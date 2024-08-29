//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html
#include "MooseAppTestApp.h"
#include "MooseAppApp.h"
#include "Moose.h"
#include "AppFactory.h"
#include "MooseSyntax.h"

InputParameters
MooseAppTestApp::validParams()
{
  InputParameters params = MooseAppApp::validParams();
  params.set<bool>("use_legacy_material_output") = false;
  params.set<bool>("use_legacy_initial_residual_evaluation_behavior") = false;
  return params;
}

MooseAppTestApp::MooseAppTestApp(InputParameters parameters) : MooseApp(parameters)
{
  MooseAppTestApp::registerAll(
      _factory, _action_factory, _syntax, getParam<bool>("allow_test_objects"));
}

MooseAppTestApp::~MooseAppTestApp() {}

void
MooseAppTestApp::registerAll(Factory & f, ActionFactory & af, Syntax & s, bool use_test_objs)
{
  MooseAppApp::registerAll(f, af, s);
  if (use_test_objs)
  {
    Registry::registerObjectsTo(f, {"MooseAppTestApp"});
    Registry::registerActionsTo(af, {"MooseAppTestApp"});
  }
}

void
MooseAppTestApp::registerApps()
{
  registerApp(MooseAppApp);
  registerApp(MooseAppTestApp);
}

/***************************************************************************************************
 *********************** Dynamic Library Entry Points - DO NOT MODIFY ******************************
 **************************************************************************************************/
// External entry point for dynamic application loading
extern "C" void
MooseAppTestApp__registerAll(Factory & f, ActionFactory & af, Syntax & s)
{
  MooseAppTestApp::registerAll(f, af, s);
}
extern "C" void
MooseAppTestApp__registerApps()
{
  MooseAppTestApp::registerApps();
}
