// sirius_test_askpass: ssh's askpass helper for tests/test_app_cluster.cpp, the
// way sirius-app and sirius-cli are it themselves (core/remote_host.hpp): it
// hands its one argument, the prompt, to the AskpassServer named by its
// environment and prints the answer it gets back.

#include "core/remote_host.hpp"

int main(int argc, char** argv) { return sirius::app::ssh::askpassMain(argc, argv); }
