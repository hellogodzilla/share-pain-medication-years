"""Run every analysis in turn and write the results to results/ (or --out).

The case-mix bootstrap takes about 15 minutes on a recent laptop; the other steps take seconds.
"""

import casemix
import decomposition
import describe
import projection
import sensitivity
import strata
import sullivan
import transitions
import trend

for step in [describe, sullivan, decomposition, trend, projection, strata, sensitivity, transitions, casemix]:
    print(f"\n--- {step.__name__}.py")
    step.main()
