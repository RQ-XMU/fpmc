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


    def predict_next(self, session_id, input_item_id):

        out = self.instance.recommendations(input_item_id, session_id)

        return pd.Series(data=out, index=range(len(out)))#这里记得改