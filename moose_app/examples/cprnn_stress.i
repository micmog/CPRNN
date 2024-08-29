[GlobalParams]
  displacements = 'disp_x disp_y disp_z'
[]

[Mesh]
  [generated]
    type = GeneratedMeshGenerator
    dim = 3
    nx = 2
    ny = 2
    nz = 2
    xmax = 1
    ymax = 1
    zmax = 1
  []
[]

[Physics/SolidMechanics/QuasiStatic]
  [all]
    strain = FINITE
    add_variables = true
    generate_output = 'stress_yy strain_yy'
  []
[]

[Materials]
  [elasticity_tensor]
    type = ComputeIsotropicElasticityTensor
    youngs_modulus = 1e9
    poissons_ratio = 0.3
  []
  # [stress]
  #   type = ComputeFiniteStrainElasticStress
  # []
  [stress]
    type = ComputeFiniteStrainRNNStress
    model_path = /Users/xg3401/projects/RNN_surrogate/compiled_model/traced_CPRNN.pt 
    model_state_path = /Users/xg3401/projects/RNN_surrogate/compiled_model/traced_CPRNN_state.pt
  []
[]

[BCs]
  [left]
    type = DirichletBC
    variable = disp_x
    boundary = left
    value = 0.0
  []
  [back]
    type = DirichletBC
    variable = disp_z
    boundary = back
    value = 0.0
  []
  [bottom]
    type = DirichletBC
    variable = disp_y
    boundary = bottom
    value = 0.0
  []
  [top]
    type = FunctionDirichletBC
    variable = disp_y
    boundary = top
    function = '0.001*t'
  []
[]

[Executioner]
  type = Transient
  end_time = 100
  dt = 5

  solve_type = 'PJFNK'
[]

[Postprocessors]
  [ave_stress]
    type = ElementAverageValue
    variable = stress_yy
  []
  [ave_strain]
    type = ElementAverageValue
    variable = strain_yy
  []
[]

[Outputs]
  exodus = true
  # perf_graph = true
  csv = true
  print_linear_residuals = false
[]