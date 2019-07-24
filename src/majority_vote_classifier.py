import linear_classifier
from data_handling import *

if __name__ == "__main__":
    # hard coded user to use.
    uuid = '1155FF54-63D3-4AB2-9863-8385D0BD0A13';
    (X, Y, M, timestamps, feature_names, label_names) = read_user_data(uuid);

    sensors_to_use = ['Acc', 'WAcc'];
    target_label = 'FIX_walking';
    feat_sensor_names = get_sensor_names_from_features(feature_names);
    model = linear_classifier.train_model(X, Y, M, feat_sensor_names, label_names, sensors_to_use, target_label);
    print("done")

    # split into real data
    linear_classifier.test_model(X, Y, M, timestamps, feat_sensor_names, label_names, model);

   # linear_classifier.test_model()
#    models = []
 #   for label_name in label_names:
  #      model = linear_classifier.train_model(X, Y, M, feature_names,label_names, ['Acc','WAcc', 'raw_acc:magnitude_stats:mean'], label_name)