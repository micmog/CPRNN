//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#pragma once

#include "ComputeStressBase.h"
// #include "GuaranteeConsumer.h"

#include <torch/script.h>

#include <iostream>
#include <memory>
#include <tuple>

using namespace torch::indexing;

/**
 * Compute stress using RNN surrogate model for finite strains
 */
class ComputeFiniteStrainRNNStress : public ComputeStressBase //, public GuaranteeConsumer
{
public:
  static InputParameters validParams();

  ComputeFiniteStrainRNNStress(const InputParameters & parameters);

  void initialSetup() override;
  virtual void initQpStatefulProperties() override;

protected:
  virtual void computeQpStress() override;

  torch::jit::script::Module load_model(std::string path);

  torch::Tensor load_tensor(std::string path);

  void save_tensor(torch::Tensor tensor, std::string path);

  torch::Tensor scale_data(torch::Tensor data, Slice idx = Slice());

  torch::Tensor scale_inputs(torch::Tensor data);

  torch::Tensor scale_outputs(torch::Tensor data);

  torch::Tensor recover_data(torch::Tensor scaled_data, Slice idx = Slice());

  torch::Tensor recover_inputs(torch::Tensor scaled_data);

  torch::Tensor recover_outputs(torch::Tensor scaled_data);

  torch::Tensor convertRankTwoTensor(const RankTwoTensor & tensor_in);
  RankTwoTensor convertRankTwoTensor(torch::Tensor tensor_in, bool sym);

  torch::Tensor convertState(std::vector<float> state_in);

  std::vector<float> convertState(torch::Tensor state_in);

  torch::jit::script::Module _torch_model, _torch_model_state;

  torch::Tensor _data_minvals, _data_maxvals;
  int _s_in_size, _s_ou_size;
  int _s_hs_size, _s_n_layer;

  /// Name of the elasticity tensor material property
  const std::string _elasticity_tensor_name;
  /// Elasticity tensor material property
  const MaterialProperty<RankFourTensor> & _elasticity_tensor;

  const MaterialProperty<RankTwoTensor> & _deformation_gradient;
  MaterialProperty<std::vector<float>> & _deformed_state;
  const MaterialProperty<std::vector<float>> & _deformed_state_old;

};
