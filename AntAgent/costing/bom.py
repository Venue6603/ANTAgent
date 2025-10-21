from typing import List
from ..models import BOMSpec, BOMTable, BOMItem

def build_bom(spec: BOMSpec) -> BOMTable:
    items: List[BOMItem] = []
    for name in spec.targets or []:
        items.append(BOMItem(name=name, vendor="Generic", catalog_no=None, unit_cost=10.0, qty=1))
    total = sum((i.unit_cost or 0.0) * i.qty for i in items)
    return BOMTable(items=items, est_total_cost=total)
