"""A stand-in for app/python/sirius_worker in the worker launcher's tests
(tests/test_app_local_worker.cpp). It never serves: it only fails to start in
the ways the real worker can, so the launcher's classification of those
failures is tested against a real process. Standard library only."""
