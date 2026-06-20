"""The two tasks.

    hud eval tasks.py claude --task-ids one_deal_ai_workflow_business -y --max-steps 8
    hud eval tasks.py claude --task-ids save_the_account_routing_bug  -y --max-steps 12
"""

from env import env, one_deal, save_the_account  # noqa: F401  (re-export env for `hud eval tasks.py`)

# Win a new customer: truthful offer -> verifiable deliverable.
_one_deal = one_deal(
    prompt=(
        "You run a tiny simulated AI workflow business. One synthetic customer has sent a "
        "request. Read the request and your company capabilities. Make a truthful offer, then "
        "submit a deliverable. You earn fake credits only if the customer accepts the offer and "
        "the deliverable passes verification."
    ),
)
_one_deal.slug = "one_deal_ai_workflow_business"

# Keep an existing customer: diagnose, fix, communicate.
_save_the_account = save_the_account(
    prompt=(
        "You run a tiny simulated AI workflow business and an existing customer is about to "
        "cancel. Read their message and the recent usage logs, then read the workflow config that "
        "is running in production. Find the routing bug, submit a corrected config with submit_fix, "
        "and send the customer an honest reply explaining what went wrong and what you changed. "
        "Do not promise anything you cannot deliver."
    ),
)
_save_the_account.slug = "save_the_account_routing_bug"

tasks = [_one_deal, _save_the_account]
