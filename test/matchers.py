from hamcrest import equal_to
from hamcrest.core.base_matcher import BaseMatcher


class HasItemsInRelativeOrder(BaseMatcher):
    def __init__(self, *matchers):
        self.matchers = [m if hasattr(m, 'matches') else equal_to(m)
                         for m in matchers]

    def _matches(self, item):
        it = iter(item)
        for m in self.matchers:
            try:
                while not m.matches(it.next):
                    pass
            except StopIteration:
                return False
        return True

    def describe_to(self, description):
        description.append_text('sequence containing in relative order [')
        first = True
        for m in self.matchers:
            if not first:
                description.append_text(', ')
            first = False
            m.describe_to(description)
        description.append_text(']')


def has_items_in_relative_order(*matchers):
    return HasItemsInRelativeOrder(*matchers)
