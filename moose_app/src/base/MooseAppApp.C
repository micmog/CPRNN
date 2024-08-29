#include "MooseAppApp.h"
#include "Moose.h"
#include "AppFactory.h"
#include "ModulesApp.h"
#include "MooseSyntax.h"

InputParameters
MooseAppApp::validParams()
{
  InputParameters params = MooseApp::validParams();
  params.set<bool>("use_legacy_material_output") = false;
  params.set<bool>("use_legacy_initial_residual_evaluation_behavior") = false;
  return params;
}

MooseAppApp::MooseAppApp(InputParameters parameters) : MooseApp(parameters)
{
  MooseAppApp::registerAll(_factory, _action_factory, _syntax);
}

MooseAppApp::~MooseAppApp() {}

void
MooseAppApp::registerAll(Factory & f, ActionFactory & af, Syntax & s)
{
  ModulesApp::registerAllObjects<MooseAppApp>(f, af, s);
  Registry::registerObjectsTo(f, {"MooseAppApp"});
  Registry::registerActionsTo(af, {"MooseAppApp"});

  /* register custom execute flags, action syntax, etc. here */
}

void
MooseAppApp::registerApps()
{
  registerApp(MooseAppApp);
}

/***************************************************************************************************
 *********************** Dynamic Library Entry Points - DO NOT MODIFY ******************************
 **************************************************************************************************/
extern "C" void
MooseAppApp__registerAll(Factory & f, ActionFactory & af, Syntax & s)
{
  MooseAppApp::registerAll(f, af, s);
}
extern "C" void
MooseAppApp__registerApps()
{
  MooseAppApp::registerApps();
}
