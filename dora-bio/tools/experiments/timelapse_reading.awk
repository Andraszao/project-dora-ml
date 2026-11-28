# Brick 40's pre-registered adoption rule (PATHWAY.md, "Brick 40 pre-registration"), applied to the summary that
# tools/experiments/timelapse.sh writes. Gate 40 applies it to the summary recorded in docs/RESULTS.md and checks the
# recorded reading and the shipped default against it.
#
#   run <movie> <block> <arm> <PSNR> <SSIM> <dynamic> <steady> <slope> <ratio-err> <found> <real> <converged> <train-min>
#
# An arm qualifies if, against A1 on BOTH movies' test blocks: PSNR >= A1 + 0.5 dB, dynamic-pixel PSNR >= A1 - 0.3 dB,
# slope >= A1's - 0.03. Of the qualifiers, the highest mean test PSNR over the two movies wins; within 0.1 dB of it,
# the one with fewer input frames (A2 1, A4 2, A3 3). The late block and the post-hoc long A3 run are never read. The
# summary holds 4-decimal values, so a margin of 1e-9 keeps an exact tie with a floor (e.g. +0.5000) on the passing side.
$1 == "run" && $3 == "test" && $4 ~ /^A[1-4]$/ {
  if (!($2 in movie)) { movie[$2] = 1; order[++movies] = $2 }   # movies in summary order, so the output is stable
  psnr[$2, $4] = $5; dyn[$2, $4] = $7; slope[$2, $4] = $9; conv[$2, $4] = $13
}
END {
  frames["A2"] = 1; frames["A4"] = 2; frames["A3"] = 3
  if (movies != 2) { print "reading: expected two movies, found " movies + 0; exit 1 }
  best = ""; bestMean = -1e9
  split("A2 A3 A4", arms, " ")
  for (k = 1; k <= 3; k++) {
    a = arms[k]; ok = 1; mean = 0; why = ""
    for (j = 1; j <= movies; j++) {
      m = order[j]
      if (!((m, a) in psnr) || !((m, "A1") in psnr)) { print "reading: missing " a " or A1 on " m; exit 1 }
      gain = psnr[m, a] - psnr[m, "A1"]; dgain = dyn[m, a] - dyn[m, "A1"]; sgain = slope[m, a] - slope[m, "A1"]
      if (gain < 0.5 - 1e-9) { ok = 0; why = why sprintf(" %s PSNR %+.4f < +0.5;", m, gain) }
      if (dgain < -0.3 - 1e-9) { ok = 0; why = why sprintf(" %s dynamic %+.4f < -0.3;", m, dgain) }
      if (sgain < -0.03 - 1e-9) { ok = 0; why = why sprintf(" %s slope %+.4f < -0.03;", m, sgain) }
      mean += psnr[m, a] / 2
    }
    printf "arm %s: %s%s (mean test PSNR %.4f)\n", a, ok ? "qualifies" : "fails", ok ? "" : ":" why, mean
    if (ok) { qual[a] = mean; if (mean > bestMean) { bestMean = mean; best = a } }
  }
  if (best == "") { print "adopted: none (the time-lapse modes stay opt-in)"; exit 0 }
  pick = best
  for (k = 1; k <= 3; k++) { a = arms[k]; if ((a in qual) && qual[a] >= bestMean - 0.1 && frames[a] < frames[pick]) pick = a }
  print "adopted: " pick
}
