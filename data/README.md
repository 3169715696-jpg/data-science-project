# Data

The AnAge extract (`anage_data.csv`, 4,218 records) is **not** redistributed in this repository.
Download it yourself from the source and place it here:

1. Open the AnAge species page: https://genomics.senescence.info/species/
2. Download the species export (the CSV used in CISC7204 Tutorial 07 is the same file).
3. Save it as `data/anage_data.csv`.

Required columns: `Class`, `Common name`, `Body mass (g)`, `Maximum longevity (yrs)`.

Then run from the repository root:

```bash
python anage_analysis.py
```

Or point the script at any other location:

```bash
python anage_analysis.py --data "C:/somewhere/anage_data.csv"
```

Cite the source (Tacutu et al., AnAge, Nucleic Acids Research) if you reuse the data or the results,
and respect the terms of use stated on the AnAge site.
