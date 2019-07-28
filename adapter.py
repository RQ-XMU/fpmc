import numpy as np
import pandas as pd
import fpmc as fpmc

class Adapter:

    def __init__(self, epochs=10, session_key='SessionId', item_key='ItemId'):
        self.algo = fpmc
        self.instance = fpmc.FPMC()
        self.epochs = epochs
        self.item_key = item_key
        self.session_key = session_key
        self.current_session = None

    def fit(self, data):
        max_iterations = ( len(data) - data.SessionId.nunique() ) * self.epochs
        progress = ( len(data) - data.SessionId.nunique() )
        min_iterations = ( len(data) - data.SessionId.nunique() )

        self.instance.prepare_model(data)
        self.instance.train(data, max_iter=max_iterations, min_iter=min_iterations, progress=progress)

    def predict_next(self, session_id, input_item_id, predict_for_item_ids, skip=False, type='view', timestamp=0):
        iidx = self.instance.item_map[input_item_id]
        if self.current_session is None or self.current_session != session_id:
            self.current_session = session_id
            self.session = [iidx]
        else:
            self.session.append(iidx)

        out = self.instance.recommendations([[iidx]], session=self.session)

        return pd.Series(data=out, index=self.instance.item_list)