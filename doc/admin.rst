Administrative Command-Line Tooling
===================================

These commands can be used to inspect, reformat, convert, and prepare a dataset for testing.
We are explicitly noting them as administrative commands to allow you to see how the system works but are not promising that the interfaces for these commands will not change or the output.
For example, the ``inspect`` subcommand is currently the bare minimum and the intent is for it to provide more information later on.
With that in mind, using these commands in operational scripts should be considered a risk.
Please contact us if there is functionality you rely on that you would like to have a stronger backwards compatibility contract.

.. contents:: Subcommands
    :local:
    :depth: 3

.. click:: lsst.images.cli:main
    :prog: lsst-images-admin
    :nested: full
