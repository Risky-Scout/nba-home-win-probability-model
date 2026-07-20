# Model card

## Model

Pre-March-selected probability model for NBA home-win outcomes using team-level
Elo, Bradley-Terry, and trend features.

## Intended use

Research / interview prototype for pregame probability estimation.

## Out of scope

- Live in-play pricing
- Customer-facing odds with overround
- Claims of market edge or closing-line value
- Production deployment without monitoring and market integration

## Training data

One NBA season file provided for the assignment (2025–26 in-file dates).

## Selection data

Games strictly before 2026-03-01.

## Evaluation

- March: locked test
- April frozen March 31: primary assignment result
- April sequential: sensitivity

## April status

April is the assignment’s retrospective scoring period. The executable
selection pipeline uses zero April rows, but April had previously been viewed
during the broader project, so I do not claim that it is a pristine untouched
holdout.

## Ethical / commercial note

Not a betting tip sheet. Not evidence of profitable wagering.
