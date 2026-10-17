Auftrag: Produktionsänderungen von FAST6a4ae34 auf Strata v0.1.45 portieren und isoliert prüfen.
Erledigt: Upstream477ac13 ausgecheckt; Produktionsdelta gegen ed4773e inklusive nativer Graphbasis dreiseitig angewendet. Drei Konfliktdateien: QSA-Header, MTP, native_mmvq. Produktion unverändert.
Naechster Schritt: Konflikte auflösen, Verhalten des Servers erhalten, bauen und testen.
Erledigt: Drei Konflikte zusammengeführt: beide QSA-APIs erhalten, beide Q6-Planarlayouts beim Draft-Gather berücksichtigt, upstream Q6-SWAR für Wide übernommen. Lokaler Coop-Pfad und FAST-Schalter bleiben.
Naechster Schritt: Build mit Produktions-CMake-Optionen; Serververhalten explizit erhalten.
Erledigt: CMake mit produktionsgleichen Optionen konfiguriert, SYCL-Build läuft. Opt-in legacy_generation_defaults erhält alte Samplingdefaults, answer_room und Anthropic-Budgetsemantik; upstream-Defaults bleiben sonst erhalten.
Naechster Schritt: Build-/Python-Ergebnisse prüfen.
Erledigt: SYCL-Engine und drei FAST-Paritätsprogramme erfolgreich gebaut (Intel icx/icpx2026.1, JIT wie Produktion);346Python-Tests bestanden. Kandidatenkonfiguration in validation/candidate.private.json mit legacy_generation_defaults=true.
Naechster Schritt: begrenzte B70-Logitparität gegen unveränderliches Produktionsbinary; danach identischer Decode-Workload und Kurzprefill-Vergleich, jeweils Restore.
