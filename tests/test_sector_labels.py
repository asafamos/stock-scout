from core.trading.config import TradingConfig
from core.trading.sector_labels import canonical_sector, expand_blocklist, same_sector


def test_synonyms_are_expanded_in_the_blocklist():
    c = TradingConfig(); c.blocked_sectors = "Consumer Defensive,Utilities,Communication,Materials,Basic Materials,Real Estate"
    b = {x.lower() for x in c.blocked_sectors_list}
    assert "consumer staples" in b                                   # the real bug: used to escape the Consumer Defensive block
    assert {"materials", "basic materials", "utilities", "real estate", "communication"} <= b


def test_communication_vs_communication_services_is_NOT_unified():
    # owner policy decision pending: blocked 'Communication' must not silently start blocking 'Communication Services'
    c = TradingConfig(); c.blocked_sectors = "Communication"
    assert "communication services" not in {x.lower() for x in c.blocked_sectors_list}
    assert canonical_sector("Communication Services") == "Communication Services"


def test_unblocked_sectors_stay_unblocked():
    c = TradingConfig(); c.blocked_sectors = "Consumer Defensive"
    assert not ({"financial", "financial services", "consumer cyclical", "consumer discretionary", "technology"} & {x.lower() for x in c.blocked_sectors_list})


def test_same_sector_uses_synonyms_and_ignores_empty():
    assert same_sector("Consumer Staples", "consumer defensive") and same_sector("Health Care", "Healthcare")
    assert not same_sector("", "") and not same_sector("Energy", "Technology")
    assert expand_blocklist(["Energy"]) == ["Energy"]
