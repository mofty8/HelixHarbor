### HelixHarbor User Manual

Welcome to **HelixHarbor**, a tool for analyzing Transmembrane Helices (TMHs) in transmembrane proteins from *Homo sapiens* and *S. cerevisiae*. This manual will guide you through the various functionalities offered by HelixHarbor.

---

### Getting Started

HelixHarbor provides four primary analysis options:

1. **Analyze an Amino Acid Sequence**
   - **Instructions:**
     1. Select the ‘Sequence’ checkbox.
     2. Paste your amino acid sequence into the provided field.
     3. Press ‘Submit’.
     4. Optionally, click on ‘Paste Example Sequence’ to use an example sequence.
   - **Output:**  Annotation of your sequence, including:
     - Position of loops and TMHs
     - Orientation (inside/outside) of the loops
     - Average surface area, volume, bulkiness, and hydrophobicity of each segment

2. **Analyze a List of Transmembrane Proteins Against Background Sets**
   - **Instructions:**
     1. Select the ‘List Against Background’ checkbox.
     2. Paste a list of UniProt IDs, separated by commas, into the provided field.
     3. Optionally, click ‘Paste Example List’ to use an example list.
     4. Choose the organism for the background set (*Homo sapiens* or *S. cerevisiae*).
     5. Select the background set (UniProt, TopDB, TmAlpha).
     6. Decide whether to include only the first helix or all helices in the comparison.
     7. Choose the physicochemical feature for comparison (Surface Area, Volume, Bulkiness, Hydrophobicity).
   - **Output:** Density plot comparing your list of interest against the background set based on the selected physicochemical feature. For the AAC feature, 20 density plots (one for each amino acid) will be generated.
   - **Custom Scale Feature:** This feature allows you to use your own scale by entering a value for each amino acid.
     1. Download the custom scale sheet.
     2. Fill in the values for each amino acid in the downloaded Excel sheet.
     3. Upload the completed custom scale sheet.
     4. This will return a density plot comparing your list of interest with the chosen background set.

3. **Compare Two Lists of Transmembrane Proteins**
   - **Instructions:**
     1. Select the ‘Compare Two Lists’ checkbox.
     2. Paste two lists of UniProt IDs, separated by commas, into the respective fields.
     3. Decide whether to include only the first helix or all helices in the comparison.
     4. Select the physicochemical feature for comparison.
   - **Output:** Density plot comparing the selected feature between the two lists.

4. **Analyze the Position-Specific Amino Acid Composition of TMHs**
   - **Instructions:**
     1. Select the ‘TMH Position Specific AAC’ checkbox.
     2. Paste your list of UniProt IDs into the provided field.
     3. Select the orientation of the helices (inside/outside). If you choose "inside," all helices transitioning from outside to inside will be reversed. Similarly, if you choose "outside," helices transitioning from inside to outside will be reversed. 
     4. Press ‘Submit’.
   - **Output:** Heatmap describing the amino acid composition of the TMHs in your list of interest.


### Background Dataset Compilation

We gathered three datasets from distinct sources: TOPDB (Topology Database of Transmembrane Proteins), TmAlphaFold, and UniprotKB. TOPDB and TmAlphaFold already included information regarding the inside/outside orientation. For UniprotKB, we applied the positive inside rule to determine orientation. Physiochemical feature calculations were performed as detailed in the following section

   
   
### Explanation of Physicochemical Scales

**Bulkiness Index:**
- Measures the steric bulk of amino acids, reflecting how much space an amino acid occupies within a protein structure.
- Reference: Zimmerman, J. M., Eliezer, N., & Simha, R. (1968). The characterization of amino acid sequences in proteins by statistical methods. *Journal of Theoretical Biology*, 21(2), 170-201.

**Volume Index:**
- Indicates the molecular volume of amino acids, measured in cubic angstroms (Å³). Represents the space occupied by an amino acid in a folded protein structure.
- Reference: Chothia, C. (1975). Structural invariants in protein folding. *Nature*, 254(5498), 304-308.

**Surface Area Index (Theoretical and Empirical):**
- Provides the accessible surface area of amino acids.
- Reference: Tien, M. Z., Meyer, A. G., Sydykova, D. K., Spielman, S. J., & Wilke, C. O. (2013). Maximum allowed solvent accessibilities of residues in proteins. *PLoS ONE*, 8(11), e80635.

**Hydrophobicity (Kyte-Doolittle Scale):**
- Measures the hydrophobicity of amino acids.
- Reference: Kyte, J., & Doolittle, R. F. (1982). A simple method for displaying the hydropathic character of a protein. *Journal of Molecular Biology*, 157(1), 105-132.





