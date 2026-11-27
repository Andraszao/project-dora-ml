# Brick 38's pre-registered reading (PATHWAY.md, commit f91afa8), applied to runs/scoreboard/summary.txt by
# tools/experiments/scoreboard.sh and to the recorded summary by gate 38. Input lines used:
#   set <name> <noisy> <bilateral> <DORA> <CAREamics N2V> <CAREamics N2V2> <five SSIMs>   (PSNR in dB, unrounded)
# - DORA is competitive on a set when its PSNR is at least the better CAREamics variant's - 0.3 dB.
# - N2V2 beating Noise2Void by >= 0.3 dB on at least two of the five sets with references (every set but CARE 2D,
#   whose floor Q20 removed) keeps stage 2's N2V2 brick next; otherwise Noise2Noise goes first.
# - DORA trailing by more than 0.3 dB on at least two of those five sets makes the largest such gap the next pass.
$1 == "set" { name[++k] = $2; dora[$2] = $5; n2v[$2] = $6; n2v2[$2] = $7 }
END {
  for (i = 1; i <= k; i++) {
    s = name[i]; best = n2v[s] > n2v2[s] ? n2v[s] : n2v2[s]; gap = best - dora[s]
    verdict = dora[s] >= best - 0.3 - 1e-9 ? "competitive" : "trails"
    printf "set %s: DORA %.2f, better CAREamics %.2f (%s), DORA %+.2f dB: %s%s\n", s, dora[s], best, (n2v[s] >= n2v2[s] ? "N2V" : "N2V2"), \
      -gap, verdict, (s == "zerocost-care2d" ? " (reported only, Q20)" : "")
    if (s == "zerocost-care2d") continue
    counted++
    if (n2v2[s] >= n2v[s] + 0.3 - 1e-9) n2v2wins++
    if (verdict == "trails") { trails++; if (gap > worst) { worst = gap; worstset = s } }
  }
  printf "sets counted: %d\n", counted
  printf "N2V2 beats Noise2Void by >= 0.3 dB on %d of %d: %s\n", n2v2wins, counted, \
    (n2v2wins >= 2 ? "next-brick: n2v2 (stage 2 order kept)" : "next-brick: noise2noise (goes first)")
  printf "DORA trails the better CAREamics variant by > 0.3 dB on %d of %d: %s\n", trails, counted, \
    (trails >= 2 ? sprintf("next-pass: %s (gap %.2f dB)", worstset, worst) : "next-pass: none (no targeted pass)")
}
