# ReSpecTh RKD fixtures

Five unmodified member files of the ReSpecTh Kinetics Data archives pinned in
`carmel/data/respecth_manifest.json`, copied byte-for-byte out of the zips (see
`.gitattributes`: line endings are never normalized, because each file is pinned by the
sha256 of its exact bytes).

Source: OSF mirror of the ReSpecTh database, https://osf.io/nbmzv/
(DOI 10.17605/OSF.IO/NBMZV), (c) RESPECTH, CC BY 4.0. Attribution and the papers the
maintainers ask users to cite are in the repository `NOTICE`.

| File | Archive (OSF file id, v1) | Member sha256 | Why it is here |
|---|---|---|---|
| `x00000070_p.xml` | `H2_indirect_v2_3.zip` (`6716900181a7f78563a5e911`) | `e5fbc013960dec9e76b28c6476d7f30a3516f300ac7a11e3dc68280d94967314` | Shock tube, reflected shock, 3 points; its comment marks the referenceDOI as a sorting placeholder |
| `x10000001.xml` | `H2_indirect_v2_3.zip` (`6716900181a7f78563a5e911`) | `f9a7f6703bff4a298d9fa99ff0e3db5d2550ce84946e3e37d3108ecf278af901` | Shock tube, reflected shock, 7 points, cited referenceDOI |
| `x10000030_x.xml` | `H2_indirect_v2_3.zip` (`6716900181a7f78563a5e911`) | `11576b4dd2559738ffc5dea6cf7456672a44896f36e569146ee05ef55bb32f86` | Shock tube with no stated mode: mapped with reflected shock ASSUMED (`mode_basis=assumed`), 7 points |
| `x40001039.xml` | `syngas_indirect_v2_3.zip` (`671691044a236f2bf52ecb24`) | `4c0066cb7e54a98326f3d85f3e0dd1e7fbad4786444b988ab33d2ffd4cb00c21` | RCM, syngas, 18 points in `mbar` at 354–377 K: its volume history compresses, so P/T are pre-compression and the record is refused (`rcm_pre_compression_conditions`) |
| `x40001058_19.xml` | `syngas_indirect_v2_3.zip` (`671691044a236f2bf52ecb24`) | `b505031a16926bc1bb22746c50636696c5c0513621779e54e78769af4017f85d` | RCM, syngas, 1 point; its volume history never falls below its first volume, so P/T are end-of-compression and the record maps |
