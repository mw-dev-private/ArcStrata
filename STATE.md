Auftrag: Produktionsänderungen von FAST6a4ae34 auf Strata v0.1.45 portieren und isoliert prüfen.
Erledigt: Upstream477ac13 ausgecheckt; Produktionsdelta gegen ed4773e inklusive nativer Graphbasis dreiseitig angewendet. Drei Konfliktdateien: QSA-Header, MTP, native_mmvq. Produktion unverändert.
Naechster Schritt: Konflikte auflösen, Verhalten des Servers erhalten, bauen und testen.
Erledigt: Drei Konflikte zusammengeführt: beide QSA-APIs erhalten, beide Q6-Planarlayouts beim Draft-Gather berücksichtigt, upstream Q6-SWAR für Wide übernommen. Lokaler Coop-Pfad und FAST-Schalter bleiben.
Naechster Schritt: Build mit Produktions-CMake-Optionen; Serververhalten explizit erhalten.
Erledigt: CMake mit produktionsgleichen Optionen konfiguriert, SYCL-Build läuft. Opt-in legacy_generation_defaults erhält alte Samplingdefaults, answer_room und Anthropic-Budgetsemantik; upstream-Defaults bleiben sonst erhalten.
Naechster Schritt: Build-/Python-Ergebnisse prüfen.
Erledigt: SYCL-Engine und drei FAST-Paritätsprogramme erfolgreich gebaut (Intel icx/icpx2026.1, JIT wie Produktion);346Python-Tests bestanden. Kandidatenkonfiguration in validation/candidate.private.json mit legacy_generation_defaults=true.
Naechster Schritt: begrenzte B70-Logitparität gegen unveränderliches Produktionsbinary; danach identischer Decode-Workload und Kurzprefill-Vergleich, jeweils Restore.
Erledigt GPU-Zwischenstand:104/1500-Token-Fälle je64Logitpositionen bit-identisch,KL0; Cache13423Slots identisch. Rohdaten work/v145-validation-20261017-113241 im Auftrag-Workspace.
Naechster Schritt:32k-Parität, dann rotierte Timingpaare; Restore folgt automatisch.
Erledigt: Alle drei Logitfälle bit-identisch (192Positionen×248320Logits),KL0.
Naechster Schritt: Rotierte32k-Decode-Timingpaare mit CPU_SHARE=auto; dann Restore und Kurzprefill-Serververgleich.

Delta Owner: Keine weiteren Vergleiche;1.45 aktivieren und laufen lassen.32k-Vergleich schon abgeschlossen, alte Produktion restauriert. Kurzprefill nur Dry-run, kein Vergleich gestartet. Deployment mit unveränderter Modell-/Env-Konfiguration und expliziten Legacy-Generationsdefaults läuft.
Naechster Schritt: Nur Health/Identität prüfen und aktive Version persistieren.
Erledigt:1.45 live, Health geladen, neuer Server-CWD/Config/Engine-Hash geprüft; Engine-Env und Originalconfig unverändert. start144/start145 wählen server_repo aus Config, Default jetzt1.45. Keine weiteren Vergleiche oder Inferenzrequests nach Owner-Delta.
Naechster Schritt: Normalbetrieb auf1.45; kein weiterer Test geplant.

Auftrag: Optionales Checkpoint-Staging portieren und B70-Gewinn prüfen. Backups im Auftrag-Workspace work/checkpoint-backup. Opt-in-Pfad mit Payload-Vergleich und Timing implementiert; Default unverändert. Nächster Schritt: Build, dann kontrollierter Vergleich mit Restore.

Erledigt: Build erfolgreich;1bd1a20. Q-Arm auf B70: vier Turn-Checkpoints bytegenau identisch zur bisherigen Sicherung, Folgeanfragen erfolgreich. Staging4,5–4,9ms; noch kein Vergleichsergebnis. ABBA-Timing läuft unter results/short_prefill_20261017_155538. Nächster Schritt: A/B abschliessen und Produktion restaurieren.

2026-10-17 Checkpoint-Staging-Test abgeschlossen:4Payloads bytegenau,48Timing-Folgefragen ABBA. Sicherung43,683->4,653ms;104-TTFT8k803,7->780,9ms,80k901,9->884,1ms, hohe Streuung/variableTokenzahl. Nachgelagerte Kopie verschiebt wahrscheinlich Arbeit zur nächsten Anfrage. Produktion0713d52 RESTORE VERIFIED. Brain checkpoint-staging-b70-test-2026-10-17.md; Bericht outputs/Checkpoint-Staging-B70.md im Auftrag-Workspace. Nächster Schritt: keine automatische Aktivierung, optionaler Kandidat bleibt aus.

2026-10-17 MessaufschlagABCCBA abgeschlossen:96Messanfragen, kein konsistenter großer Aufschlag; Rohdaten checkpoint-overhead-20261017-162236 im Auftrag-Workspace. Teilzeitpatch opt-in STRATA_CKPT_DETAIL angewendet; Build läuft. Nächster Schritt: Host-Kostenaufteilung8k, danach Restore.

2026-10-17 Detailmessung abgeschlossen und Produktion restauriert: Baseline Resize31,83ms/GPUcopy14,62ms/Keep2,52ms; Staging GPUcopy4,73ms, nach DONE Resize32,76ms+RAMcopy7,22ms+Keep2,99ms, Finish42,69ms. Upstream CheckpointPool für Single-GPU-SYCL opt-in portiert; Build und Dry-run bestanden. Nächster Schritt: Bytevergleich und balancierter Pool/Pool+Staging-Test; danach unveränderte Produktion restaurieren.
