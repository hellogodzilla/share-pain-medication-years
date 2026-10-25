# Runs every script in turn; results go to results/ (or --out).

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
