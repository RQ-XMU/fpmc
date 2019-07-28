class MRR:

    def __init__(self, length=20):
        self.length = length

    def init(self, train):
        '''
        do initialization work here
        :param train:
        :return:
        '''
        return

    def reset(self):
        self.test = 0
        self.pos = 0

    def skip(self, for_item = 0, session = -1):
        pass

    def add(self, result, next_item, for_item=0, session=0, pop_bin=None, position=None ):
        res = result[:self.length]
        self.test += 1

        if next_item in res.index:
            rank = res.index.get_loc( next_item ) + 1
            self.pos += ( 1.0/rank )

    def add_batch(self, result, next_item):
        i = 0
        for part, series in result.iteritems():
            result.sort_values( part, ascending=False, inplace=True )
            self.add( series, next_item[i] )
            i += 1

    def result(self):
        return ("MRR@" + str(self.length) + ": "), (self.pos/self.test)



class HitRate:
    def __init__(self, length=20):
        self.length = length

    def init(self, train):
        return

    def reset(self):
        self.test = 0
        self.hit = 0

    def add(self, result, next_item, for_item=0, session=0):
        self.test += 1
        if next_item in result[:self.length].index:
            self.hit += 1

    def add_batch(self, result, next_item):
        i = 0
        for part, series in result.iteritems():
            result.sort_values(part, ascending=False, inplace=True)
            self.add(series, next_item[i])
            i += 1

    def result(self):
        return ("HitRate@" + str(self.length) + ": "), (self.hit/self.test)




