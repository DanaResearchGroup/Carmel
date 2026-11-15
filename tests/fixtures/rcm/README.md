# RCM thermodynamic references

`burcat-isopentane.dat` is the four-line `C5H12,i-pentane` (P10/85) record
from [Burcat THERM.DAT](https://respecth.elte.hu/burcat/THERM.DAT), retrieved
2026-10-17. Full-source SHA-256:
`1bcc0261e5003a94496c07632a5727e312fc9055a18e61fdb7c2d5bdca8eacaa`.
The excerpt keeps the source columns, with LF line endings. Its SHA-256 is
`c7235a7bbcb3dde2a2cfb49b0f547fe30f7196bfa47cb1f3d9447fa898d209b3`.
The first fourteen coefficients define NASA7; the last field on line four
is not a polynomial coefficient.

`ic5-cantera-3.2.0-isentropic.json` contains independent Cantera 3.2.0
entropy/volume solutions. Start with the pinned `gri30.yaml` species, append
IC5H12 using the excerpt's coefficients and 200/1000/6000 K bounds, and create
an ideal-gas solution without reactions. For each input, set `TPX`, read `SV`,
then set `SV` to the same entropy and the initial specific volume divided by
the volume ratio. Save the resulting T and P. This uses no reaction mechanism
or time integration. Cases cover the 295 K allowance and both sides of the
NASA7 midpoint; the JSON records the inputs and source hashes.
