//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#include "ComputeFiniteStrainRNNStress.h"
#include "LibtorchUtils.h"

#include <torch/script.h>

#include <iostream>
#include <memory>
#include <tuple>

using namespace torch::indexing;

registerMooseObject("SolidMechanicsApp", ComputeFiniteStrainRNNStress);

InputParameters
ComputeFiniteStrainRNNStress::validParams()
{
  InputParameters params = ComputeStressBase::validParams();
  params.addClassDescription("Compute stress using RNN surrogate model for finite strains");
  // params.addParam<FileName>(
  //   "slip_sys_hard_prop_file_name",
  //   "",
  //   "Name of the file containing the values of hardness evolution parameters");
  params.addParam<std::string>("model_path", "", "Path to saved pytorch model.");
  params.addParam<std::string>("model_state_path", "", "Path to saved pytorch state projection model.");
  return params;
}

ComputeFiniteStrainRNNStress::ComputeFiniteStrainRNNStress(
    const InputParameters & parameters)
  : ComputeStressBase(parameters),
    // GuaranteeConsumer(this),
    _elasticity_tensor_name(_base_name + "elasticity_tensor"),
    _elasticity_tensor(getMaterialPropertyByName<RankFourTensor>(_elasticity_tensor_name)),
    _deformation_gradient(getMaterialProperty<RankTwoTensor>("deformation_gradient")),
    _deformed_state(declareProperty<std::vector<float>>(_base_name + "deformed_state")),
    _deformed_state_old(getMaterialPropertyOldByName<std::vector<float>>(_base_name + "deformed_state"))
{
  // Load model
  _torch_model = load_model(getParam<std::string>("model_path"));
  _torch_model_state = load_model(getParam<std::string>("model_state_path"));

  // Load parameters
  //TODO: move to input file
  _data_minvals = torch::tensor({8.3097500e-01, -2.0384243e-01, -2.0328778e-01, -2.0427726e-01,
                                 8.2604301e-01, -2.0230727e-01, -2.0864569e-01, -2.0109621e-01,
                                 8.3100885e-01, 9.9861705e-01, -8.4226968e+07, -8.2657352e+07,
                                 -8.4674888e+07, -7.1155808e+07, -6.7956288e+07, -6.9477592e+07,
                                 -1.0477044e+08});
  _data_maxvals = torch::tensor({1.2154737e+00, 2.0595254e-01, 2.0954524e-01, 2.0795909e-01,
                                 1.2141997e+00, 2.0823230e-01, 2.0162778e-01, 2.0144145e-01,
                                 1.1949377e+00, 1.0012770e+00, 8.7337312e+07, 8.3576712e+07,
                                 8.4541888e+07, 6.9114216e+07, 6.8503744e+07, 6.9446768e+07,
                                 9.7198632e+07});
  _s_in_size = 10, _s_ou_size = 7;
  _s_hs_size = 128, _s_n_layer = 2;
}

torch::jit::script::Module ComputeFiniteStrainRNNStress::load_model(std::string path)
{
  torch::jit::script::Module model;
  try
  {
    model = torch::jit::load(path);
  }
  catch (const c10::Error &e)
  {
    std::cerr << "error loading the model at " << path << std::endl;
  }
  return model;
}

torch::Tensor ComputeFiniteStrainRNNStress::load_tensor(std::string path)
{
  std::ifstream input(path, std::ios::binary);
  std::vector<char> bytes(
      (std::istreambuf_iterator<char>(input)),
      (std::istreambuf_iterator<char>()));
  input.close();
  return torch::jit::pickle_load(bytes).toTensor();
}

void ComputeFiniteStrainRNNStress::save_tensor(torch::Tensor tensor, std::string path)
{
  auto pickled = torch::jit::pickle_save(tensor);
  std::ofstream fout(path, std::ios::out | std::ios::binary);
  fout.write(pickled.data(), pickled.size());
  fout.close();
}

torch::Tensor ComputeFiniteStrainRNNStress::scale_data(torch::Tensor data, Slice idx)
{
  torch::Tensor scaled_data = data.clone();
  scaled_data -= _data_minvals.index({idx});
  scaled_data /= _data_maxvals.index({idx}) - _data_minvals.index({idx});
  return scaled_data;
}

torch::Tensor ComputeFiniteStrainRNNStress::scale_inputs(torch::Tensor data)
{
  return scale_data(data, Slice(None, _s_in_size));
}

torch::Tensor ComputeFiniteStrainRNNStress::scale_outputs(torch::Tensor data)
{
  return scale_data(data, Slice(_s_in_size, None));
}

torch::Tensor ComputeFiniteStrainRNNStress::recover_data(torch::Tensor scaled_data, Slice idx)
{
  torch::Tensor data = scaled_data.clone();
  data *= _data_maxvals.index({idx}) - _data_minvals.index({idx});
  data += _data_minvals.index({idx});
  return data;
}

torch::Tensor ComputeFiniteStrainRNNStress::recover_inputs(torch::Tensor scaled_data)
{
  return recover_data(scaled_data, Slice(None, _s_in_size));
}

torch::Tensor ComputeFiniteStrainRNNStress::recover_outputs(torch::Tensor scaled_data)
{
  return recover_data(scaled_data, Slice(_s_in_size, None));
}

torch::Tensor ComputeFiniteStrainRNNStress::convertRankTwoTensor(const RankTwoTensor & tensor_in)
{
  return torch::tensor({
    tensor_in(0, 0), tensor_in(0, 1), tensor_in(0, 2),
    tensor_in(1, 0), tensor_in(1, 1), tensor_in(1, 2),
    tensor_in(2, 0), tensor_in(2, 1), tensor_in(2, 2)
  }).reshape({3, 3});
}

RankTwoTensor ComputeFiniteStrainRNNStress::convertRankTwoTensor(torch::Tensor tensor_in, bool sym)
{
  RankTwoTensor tensor_out(RankTwoTensor::initIdentity);
  tensor_out(0, 0) = tensor_in[0].item<Real>();
  tensor_out(1, 1) = tensor_in[1].item<Real>();
  tensor_out(2, 2) = tensor_in[2].item<Real>();
  tensor_out(0, 1) = tensor_in[3].item<Real>();
  tensor_out(1, 0) = tensor_in[3].item<Real>();
  tensor_out(0, 2) = tensor_in[4].item<Real>();
  tensor_out(2, 0) = tensor_in[4].item<Real>();
  tensor_out(1, 2) = tensor_in[5].item<Real>();
  tensor_out(2, 1) = tensor_in[5].item<Real>();
  return tensor_out;
}

std::vector<float> ComputeFiniteStrainRNNStress::convertState(torch::Tensor state_in)
{
  std::vector<float> state_out;
  state_in = state_in.flatten();
  LibtorchUtils::tensorToVector(state_in, state_out);
  return state_out;
}

torch::Tensor ComputeFiniteStrainRNNStress::convertState(std::vector<float> state_in)
{
  torch::Tensor state_out;
  LibtorchUtils::vectorToTensor(state_in, state_out, true);
  return state_out.reshape({_s_n_layer, _s_hs_size});;
}

void
ComputeFiniteStrainRNNStress::initialSetup()
{
}

void
ComputeFiniteStrainRNNStress::initQpStatefulProperties()
{
  ComputeStressBase::initQpStatefulProperties();

  torch::Tensor init_state = load_tensor("//Users//xg3401//projects//RNN_surrogate//compiled_model//init_state.pt");
  // Project initial state
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(init_state);
  torch::Tensor state = _torch_model_state.forward(inputs).toTensor();
  state = state.reshape({_s_hs_size, _s_n_layer}).transpose(0, 1);

  _deformed_state[_qp] = convertState(state);
}

void
ComputeFiniteStrainRNNStress::computeQpStress()
{

  // std::cout << "blafdhasdkjbasd" << std::endl;

  // std::cout << _qp << std::endl;
  // std::cout << std::endl;

  // std::cout << _elasticity_tensor[_qp] << std::endl;
  // std::cout << std::endl;

  // std::cout << _mechanical_strain[_qp] << std::endl;
  // std::cout << std::endl;

  // std::cout << _deformation_gradient[_qp] << std::endl;
  // std::cout << std::endl;

  torch::Tensor F = convertRankTwoTensor(_deformation_gradient[_qp]);
  torch::Tensor X = torch::cat({F.reshape(-1), torch::det(F).unsqueeze(0)});
  X = scale_inputs(X);

  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(X);
  inputs.push_back(convertState(_deformed_state_old[_qp]));

  auto outputs = _torch_model.forward(inputs).toTuple();
  _deformed_state[_qp] = convertState(outputs->elements()[1].toTensor());
  torch::Tensor Y = recover_outputs(outputs->elements()[0].toTensor());

  _stress[_qp] = convertRankTwoTensor(Y.index({Slice(None, 6)}), true);

  // std::cout << _qp << std::endl;
  // std::cout << _deformation_gradient[_qp] << std::endl;
  // std::cout << _stress[_qp] << std::endl;
  // std::cout << std::endl;

  // Assign value for elastic strain, which is equal to the mechanical strain
  _elastic_strain[_qp] = _mechanical_strain[_qp];

  // Compute dstress_dstrain
  _Jacobian_mult[_qp] = _elasticity_tensor[_qp];

}
