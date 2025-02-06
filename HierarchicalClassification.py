import argparse
import pandas as pd
import json
import joblib
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
import warnings

warnings.filterwarnings("ignore")


class HierarchicalClassification:
    def __init__(self, model_path='./model/'):
        """
        Initialize classifier
        """
        self.model_path = model_path
        self._setup_logging()

    def _setup_logging(self):
        """Configure logs"""
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def load_and_process_data(self, file_path):
        """Conophil Gourley Logos"""
        with open(file_path, 'r') as file:
            data = [json.loads(line) for line in file]
        return pd.DataFrame(data)

    def predict_and_evaluate(self, model_name, data, features, label_column, result_column):
        """
        Load models, make predictions, calculate accuracy, and dynamically set results based on confidence
        :param model_name: The name of the model file
        :param data: Enter data
        :param features: A list of feature columns
        :param label_column: True label column
        :param result_column: Forecast Result column
        :return: The updated data
        """
        model = joblib.load(self.model_path + model_name)

        X = pd.concat([pd.DataFrame(data[f].tolist()) for f in features], axis=1)
        X.columns = range(X.shape[1])

        y = data[label_column]
        y_pred = model.predict(X)

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            max_confidence = proba.max(axis=1)
            confidence_threshold = 5 * (1 / data[label_column].nunique())
            y_pred = [pred if conf >= confidence_threshold else 'unknown' for pred, conf in zip(y_pred, max_confidence)]
        else:
            logging.warning(f"Model {model_name} does not support predict_proba. Falling back to normal predict.")

        valid_mask = [pred != 'unknown' for pred in y_pred]
        accuracy = accuracy_score(y[valid_mask], [pred for pred, valid in zip(y_pred, valid_mask) if valid])
        logging.info(f"Accuracy for {model_name}: {accuracy:.4f}")

        data[result_column] = y_pred
        return data

    def insert_prediction_columns(self, data, label_column, prediction_column, columns_to_fill):
        """
        Load the model, predict, and insert the predicted values of other columns based on the prediction results
        :param data: Enter data
        :param label_column: True label column
        :param prediction_column: Forecast result column
        :param columns_to_fill: Columns that need to be populated
        :return: The updated data
        """
        for col in columns_to_fill:
            data[f'pre_{col}'] = data.apply(
                lambda row: row[col] if row[label_column] == row[prediction_column] else 'unknown', axis=1)
        return data

    def classify_ascomycota(self, data):
        """ Ascomycota """
        ascomycota_data = self.predict_and_evaluate('Ascomycota_p2c.pkl', data, ['seq_vector10', 'backseq_vector10'],
                                                    'c', 'pre_c')
        ascomycota_groups = {c: df for c, df in ascomycota_data.groupby('pre_c')}

        results = []
        for class_label, model_name in {
            'Sordariomycetes': 'Ascomycota_p_Sordariomycetes_c2s.pkl',
            'Dothideomycetes': 'Ascomycota_p_Dothideomycetes_c2s.pkl',
            'Leotiomycetes': 'Ascomycota_p_Leotiomycetes_c2s.pkl',
            'Eurotiomycetes': 'Ascomycota_p_Eurotiomycetes_c2s.pkl',
            'Pezizomycetes': 'Ascomycota_p_Pezizomycetes_c2s.pkl',
        }.items():
            df = ascomycota_groups.get(class_label, pd.DataFrame())
            if not df.empty:
                df = self.predict_and_evaluate(model_name, df, ['seq_vector7', 'backseq_vector7'], 's', 'pre_s')
                df = self.insert_prediction_columns(df, 's', 'pre_s', ['o', 'f', 'g'])
                results.append(df)

        other_data = pd.concat([df for c, df in ascomycota_groups.items() if c not in [
            'Sordariomycetes', 'Dothideomycetes', 'Leotiomycetes', 'Eurotiomycetes', 'Pezizomycetes'
        ]])
        if not other_data.empty:
            other_data = self.predict_and_evaluate('Ascomycota_p_other_c2s.pkl', other_data,
                                                   ['seq_vector7', 'backseq_vector7'], 's', 'pre_s')
            other_data = self.insert_prediction_columns(other_data, 's', 'pre_s', ['o', 'f', 'g'])
            results.append(other_data)

        return pd.concat(results)

    def classify_basidiomycota(self, data):
        """Basidiomycota"""
        results = []

        # Step 1: p -> c
        basidiomycota_data = self.predict_and_evaluate(
            'Basidiomycota_p2c.pkl', data,
            ['seq_vector10', 'backseq_vector10'], 'c', 'pre_c'
        )
        basidiomycota_groups = basidiomycota_data.groupby('pre_c')

        # Step 2:  Agaricomycetes
        if 'Agaricomycetes' in basidiomycota_groups.groups:
            agaricomycetes_data = basidiomycota_groups.get_group('Agaricomycetes')

            # c -> o
            agaricomycetes_data = self.predict_and_evaluate(
                'Basidiomycota_p_Agaricomycetes_c2o.pkl', agaricomycetes_data,
                ['seq_vector10', 'backseq_vector10'], 'o', 'pre_o'
            )
            agaricomycetes_orders = agaricomycetes_data.groupby('pre_o')

            # （Orders）
            for order_name, order_data in agaricomycetes_orders:
                if order_name == 'Agaricales':
                    # Step 3-1: Agaricales 需要先进行 o -> f 分类
                    order_data = self.predict_and_evaluate(
                        'Basidiomycota_p_Agaricomycetes_c_Agaricales_o2f.pkl', order_data,
                        ['seq_vector10', 'backseq_vector10'], 'f', 'pre_f'
                    )
                    families = order_data.groupby('pre_f')

                    #  Cortinariaceae
                    if 'Cortinariaceae' in families.groups:
                        cortinariaceae_data = families.get_group('Cortinariaceae')
                        cortinariaceae_data = self.predict_and_evaluate(
                            'Basidiomycota_p_Agaricomycetes_c_Agaricales_o_Cortinariaceae_f2s.pkl',
                            cortinariaceae_data, ['seq_vector7', 'backseq_vector7'], 's', 'pre_s'
                        )
                        cortinariaceae_data = self.insert_prediction_columns(cortinariaceae_data, 's', 'pre_s', ['g'])
                        results.append(cortinariaceae_data)

                    #  Inocybaceae
                    if 'Inocybaceae' in families.groups:
                        inocybaceae_data = families.get_group('Inocybaceae')
                        inocybaceae_data = self.predict_and_evaluate(
                            'Basidiomycota_p_Agaricomycetes_c_Agaricales_o_Inocybaceae_f2s.pkl',
                            inocybaceae_data, ['seq_vector7', 'backseq_vector7'], 's', 'pre_s'
                        )
                        inocybaceae_data = self.insert_prediction_columns(inocybaceae_data, 's', 'pre_s', ['g'])
                        results.append(inocybaceae_data)

                    # Step 3-2: other1, other2
                    other1_families = []
                    other2_families = []
                    for family_name, family_data in families:
                        if family_name not in ['Cortinariaceae', 'Inocybaceae']:
                            if family_name in [
                                'Hydnangiaceae', 'Hymenogastraceae', 'Amanitaceae',
                                'Clavariaceae', 'Mycenaceae', 'Tricholomataceae',
                                'Physalacriaceae', 'Omphalotaceae', 'Hygrophoraceae'
                            ]:
                                other1_families.append(family_data)
                            else:
                                other2_families.append(family_data)

                    #  other1
                    if other1_families:
                        other1_data = pd.concat(other1_families)
                        other1_data = self.predict_and_evaluate(
                            'Basidiomycota_p_Agaricomycetes_c_Agaricales_o_other1_f2s.pkl',
                            other1_data, ['seq_vector7', 'backseq_vector7'], 's', 'pre_s'
                        )
                        other1_data = self.insert_prediction_columns(other1_data, 's', 'pre_s', ['g'])
                        results.append(other1_data)

                    #  other2
                    if other2_families:
                        other2_data = pd.concat(other2_families)
                        other2_data = self.predict_and_evaluate(
                            'Basidiomycota_p_Agaricomycetes_c_Agaricales_o_other2_f2s.pkl',
                            other2_data, ['seq_vector7', 'backseq_vector7'], 's', 'pre_s'
                        )
                        other2_data = self.insert_prediction_columns(other2_data, 's', 'pre_s', ['g'])
                        results.append(other2_data)

                else:
                    # Step 3-2:  o -> s
                    if order_name in ['Russulales', 'Sebacinales', 'Cantharellales']:
                        model_name = f'Basidiomycota_p_Agaricomycetes_c_{order_name}_o2s.pkl'
                    else:
                        # other_o2s.pkl
                        continue

                    if not order_data.empty:
                        order_data = self.predict_and_evaluate(
                            model_name, order_data,
                            ['seq_vector7', 'backseq_vector7'], 's', 'pre_s'
                        )
                        order_data = self.insert_prediction_columns(order_data, 's', 'pre_s', ['f', 'g'])
                        results.append(order_data)

            # Step 4: ~Agaricomycetes
            other_orders_data = []
            for order_name, order_data in agaricomycetes_orders:
                if order_name not in ['Agaricales', 'Russulales', 'Sebacinales', 'Cantharellales']:
                    other_orders_data.append(order_data)

            if other_orders_data:
                other_orders_data = pd.concat(other_orders_data)
                other_orders_data = self.predict_and_evaluate(
                    'Basidiomycota_p_Agaricomycetes_c_other_o2s.pkl', other_orders_data,
                    ['seq_vector7', 'backseq_vector7'], 's', 'pre_s'
                )
                other_orders_data = self.insert_prediction_columns(other_orders_data, 's', 'pre_s', ['f', 'g'])
                results.append(other_orders_data)

        # Step 5: ~Agaricomycetes
        other_basidiomycota = []
        for group_name, group_data in basidiomycota_groups:
            if group_name != 'Agaricomycetes':
                other_basidiomycota.append(group_data)

        if other_basidiomycota:
            other_basidiomycota = pd.concat(other_basidiomycota)
            other_basidiomycota = self.predict_and_evaluate(
                'Basidiomycota_p_other_c2s.pkl', other_basidiomycota,
                ['seq_vector7', 'backseq_vector7'], 's', 'pre_s'
            )
            other_basidiomycota = self.insert_prediction_columns(other_basidiomycota, 's', 'pre_s', ['o', 'f', 'g'])
            results.append(other_basidiomycota)

        return pd.concat(results) if results else pd.DataFrame()

    def classify(self, data_path):

        data = self.load_and_process_data(data_path)

        # k2p
        data = self.predict_and_evaluate('k2p.pkl', data, ['seq_vector10', 'backseq_vector10'], 'p', 'pre_p')

        # data_group
        partitioned_data = {p: df for p, df in data.groupby('pre_p')}
        basidiomycota_data = partitioned_data.get('Basidiomycota', pd.DataFrame())
        ascomycota_data = partitioned_data.get('Ascomycota', pd.DataFrame())
        other_data = pd.concat([df for p, df in partitioned_data.items() if p not in ['Basidiomycota', 'Ascomycota']])

        # other_p2s
        if not other_data.empty:
            other_data = self.predict_and_evaluate('other_p2s.pkl', other_data, ['seq_vector7', 'backseq_vector7'], 's',
                                                   'pre_s')
            other_data = self.insert_prediction_columns(other_data, 's', 'pre_s', ['c', 'o', 'f', 'g'])

        #  Ascomycota
        if not ascomycota_data.empty:
            ascomycota_data = self.classify_ascomycota(ascomycota_data)

        #  Basidiomycota
        if not basidiomycota_data.empty:
            basidiomycota_data = self.classify_basidiomycota(basidiomycota_data)

        # result_concat
        result = pd.concat([other_data, ascomycota_data, basidiomycota_data], axis=0).sort_index()
        result.drop(columns=['seq_vector7', 'backseq_vector7', 'seq_vector10', 'backseq_vector10'], inplace=True)
        result = result[
            ['seq', 'k', 'p', 'c', 'o', 'f', 'g', 's', 'pre_p', 'pre_c', 'pre_o', 'pre_f', 'pre_g', 'pre_s']]

        for col in ['p', 'c', 'o', 'f', 'g', 's']:
            accuracy = (result[col] == result[f'pre_{col}']).mean()
            logging.info(f"Overall accuracy for {col}: {accuracy:.4f}")

        return result

    def get_HA(self, df, prefixs):
        """HA (Hierarchical Accuracy)"""
        HA = 0
        for i, row in df.iterrows():
            if all(row[pre] == row[f'pre_{pre}'] for pre in prefixs):
                HA += 1
        return HA / df.shape[0]

    def get_metrics(self, df, prefixs):
        accuracy, precision, recall, f1, HA, mcc = [], [], [], [], [], []
        for i, pre in enumerate(prefixs):
            y_pred = df[f'pre_{pre}'].tolist()
            y_true = df[f'{pre}'].tolist()
            accuracy.append(accuracy_score(y_true, y_pred))
            precision.append(precision_score(y_true, y_pred, average='micro'))
            recall.append(recall_score(y_true, y_pred, average='micro'))
            f1.append(f1_score(y_true, y_pred, average='micro'))
            mcc.append(matthews_corrcoef(y_true, y_pred))
            HA.append(self.get_HA(df, prefixs[:i + 1]))

        metrics = pd.DataFrame({
            'levels': prefixs,
            'Accuracy': accuracy,
            'HA': HA,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'MCC': mcc,
        })

        return metrics

def main():
    # Set command line parameter parsing
    parser = argparse.ArgumentParser(description="Hierarchical Classification Tool")
    parser.add_argument('--classify', type=str, help="Path to the input data for classification")
    parser.add_argument('--metrics', type=str, help="Path to the classification result for metrics calculation")
    parser.add_argument('--output', type=str, default="metrics.csv", help="Path to save the metrics output")
    args = parser.parse_args()

    # Initialize the classifier
    hc = HierarchicalClassification()

    if args.classify:
        # classify
        res = hc.classify(args.classify)
        res.to_csv("classification_result.csv", index=False)
        logging.info("Classification completed. Results saved to 'classification_result.csv'.")

    if args.metrics:
        # metrics
        df = pd.read_csv(args.metrics)
        prefixs = ['p', 'c', 'o', 'f', 'g', 's']
        metrics = hc.get_metrics(df, prefixs)
        metrics.to_csv(args.output, index=False)
        logging.info(f"Metrics calculation completed. Results saved to '{args.output}'.")


if __name__ == '__main__':
    main()