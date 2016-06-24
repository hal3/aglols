import Levenshtein


class InvIndex:
    def __init__(self):
        self.word2id = {}
        self.items   = []
        self.docfreq = {}

    def add(self, item):
        i = len(self.items)
        self.items.append(item)
        for w in item.split():
            if w in self.word2id: self.word2id[w].append(i)
            else: self.word2id[w] = [i]
            self.docfreq[w] = self.docfreq.get(w,0) + 1

    def find(self, query):
        hit = set()
        for w in query.split():
            if len(w) <= 1: continue
            if w not in self.word2id: continue
            if 40 * self.docfreq.get(w,0) > len(self.items): continue
            for i in self.word2id[w]:
                if i in hit: continue
                yield self.items[i]
                hit.add(i)

    def iter(self):
        for item in self.items: yield item

def buildCanopies(invIndex, T1=0.9, T2=0.7, simFn=Levenshtein.jaro_winkler):
    assert(T2 < T1)
    allItems = list(invIndex.iter())
    removed  = set()
    allItems.sort(key=len, reverse=True)
    while len(allItems) > 0:
        item = allItems.pop()
        if item in removed: continue
        removed.add(item)
        canopy   = set([item])
        for sim in invIndex.find(item):
            if sim in removed: continue
            d = 1. - simFn(item, sim) # Levenshtein.jaro_winkler(item, sim)
            if d <= T1:
                canopy.add(sim)
            if d <= T2:
                removed.add(sim)
        yield canopy

