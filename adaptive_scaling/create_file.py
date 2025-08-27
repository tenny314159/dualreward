import os

train_num = 'train59'

if not os.path.exists('./data/%s_data' % train_num):
    os.makedirs('./data/%s_data' % train_num)

if not os.path.exists('./evaluation/%s_evaluation' % train_num):
    os.makedirs('./evaluation/%s_evaluation' % train_num)

if not os.path.exists('./model_output/%s_model' % train_num):
    os.makedirs('./model_output/%s_model' % train_num)

if not os.path.exists('./result/%s_result' % train_num):
    os.makedirs('./result/%s_result' % train_num)



