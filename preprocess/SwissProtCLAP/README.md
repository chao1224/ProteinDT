# Download Instruction

For the dataset downloads, we can check this website: https://www.uniprot.org/help/downloads.

## Reviewed Swiss-Prot
```
wget https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.xml.gz
```

The xml schema can be found [here](https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot.xsd).

## Preprocessing to get SwissProtCLAP

```
python step_01_parse_XML.py
python step_02_generate_protein_text_pair.py
```
