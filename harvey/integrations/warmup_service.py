"""Which warmup service Harvey reads: InboxWarm.ai first, TrulyInbox as a fallback.

InboxWarm (a Clustox product) is the one Harvey recommends. A TrulyInbox key
still works for anyone already on it; when both keys are set, InboxWarm wins.
"""

from harvey.integrations import inboxwarm as iw
from harvey.integrations import trulyinbox as ti


def active() -> str:
    """'inboxwarm', 'trulyinbox', or '' when no warmup key is set."""
    if iw.key_from_env():
        return "inboxwarm"
    if ti.key_from_env()[0]:
        return "trulyinbox"
    return ""


def label(service: str | None = None) -> str:
    service = active() if service is None else service
    return {"trulyinbox": "TrulyInbox"}.get(service, iw.NAME)


async def current_health(address: str):
    """(health, error, label) from whichever service is connected."""
    service = active()
    if service == "trulyinbox":
        health, error = await ti.current_health(address)
    else:
        health, error = await iw.current_health(address)
    return health, error, label(service)
