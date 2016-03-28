addpath('mnistHelper/');
data_set = PrepareTrainData();
test_set = PrepareTestData(data_set.mean, data_set.std);
net_param = InitNetParam;
net = InitNet(net_param, data_set, test_set);
clear data_set test_set;

net = TrainNet(net);




