# TARDIS-Zygon
Interactive python widget for creating and editing abundance and density profiles to run in the spectral synthesis code TARDIS.

The jupyter notebook 'abundance_editor.ipynb' explains and demonstrates the functionality of the widget. At the moment there is not checking as to whether the user entered elements are indeed elements in the TARDIS atomic data. This is currently only written to work with the 'test_abund.dat' and 'test_den.dat' style of abundance composition withing TARDIS, and not the CSVY format as of yet.

This repository served as a prototype for a Google Summer of Code project to develop a TARDIS Custom Abundance Widget: https://tardis-sn.github.io/tardis/io/visualization/using_widgets.html#custom-abundance-widget.
