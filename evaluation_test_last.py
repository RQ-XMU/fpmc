import time
import numpy as np

def evaluate_sessions(pr, metrics, test_data, train_data, items=None, session_key='SessionId', item_key='ItemId', time_key='Time'):

    actions = len(test_data)
    sessions = len(test_data)
    # sessions = len(test_data[session_key].unique())
    count = 0
    print('START evaluation of ', actions, ' actions in ', sessions, ' sessions')

    sc = time.clock()
    st = time.time()

    time_sum = 0
    time_sum_clock = 0
    time_count = 0

    for m in metrics:
        m.reset()

    # test_data.sort_values([session_key, time_key], inplace=True)
    # items_to_predict = train_data[item_key].unique()

    # prev_iid, prev_sid = -1, -1
    # iid, sid = -1, -1

    for user in range(len(test_data)):

        if count % 200 == 0:
            print('   eval process: ', count, ' of ', actions, ' actions: ', ( count / actions * 100.0), ' % in', (time.time() - st), 's')

        current_user = test_data[user][0]
        current_item = test_data[user][1][1]
        crs = time.clock()
        trs = time.time()
        preds = pr.predict_next(current_user, current_item)
        preds[np.isnan(preds)] = 0
        preds.sort_values(ascending=False, inplace=True)
        time_sum_clock += time.clock() - crs
        time_sum += time.time() - trs
        time_count += 1
        if test_data[user][2] > 1.0:#1.0评分的要用另外一种评价方式
            for m in metrics:
                m.add(preds, test_data[user][1][0])
            count += 1
    print('END evaluation in ', (time.clock() - sc), 'c / ', (time.time() - st), 's')
    print('    avg rt ', (time_sum / time_count), 's / ', (time_sum_clock / time_count), 'c')
    res = []
    for m in metrics:
        res.append(m.result())

    return res

    #     next_sid = test_data[session_key].values[i]
    #     next_iid = test_data[item_key].values[i]
    #
    #     if sid != next_sid:
    #         prev_sid = sid
    #         sid = next_sid
    #     else:
    #         if items is not None:
    #             if np.in1d(iid, items):
    #                 items_to_predict = items
    #             else:
    #                 items_to_predict = np.hstack(([iid], items))
    #
    #         if prev_iid > 0 and prev_sid >0 :
    #
    #             if prev_sid != sid:
    #
    #                 crs = time.clock()
    #                 trs = time.time()
    #
    #                 preds = pr.predict_next(prev_sid, prev_iid, items_to_predict)
    #
    #                 preds[np.isnan(preds)] = 0
    #                 preds.sort_values( ascending=False, inplace=True )
    #
    #                 time_sum_clock += time.clock() - crs
    #                 time_sum += time.time() - trs
    #                 time_count += 1
    #
    #                 for m in metrics:
    #                     m.add( preds, iid, for_item=prev_iid, session=prev_sid )
    #             else:
    #                 preds = pr.predict_next(prev_sid, prev_iid, items_to_predict, skip=True)
    #                 m.skip( for_item=prev_iid, session=prev_sid )
    #
    #         prev_sid = sid
    #         prev_iid = iid
    #         iid = next_iid
    #
    #     count += 1
    #
    # print( 'END evaluation in ', (time.clock()-sc), 'c / ', (time.time()-st), 's' )
    # print( '    avg rt ', (time_sum/time_count), 's / ', (time_sum_clock/time_count), 'c' )
    #
    # res = []
    # for m in metrics:
    #     res.append( m.result() )
    #
    # return res




