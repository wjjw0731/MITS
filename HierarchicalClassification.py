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
        Initialize the classifier
        :param model_path
        """
        self.model_path = model_path
        self._setup_logging()

    def _setup_logging(self):
        """Configure logs"""
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def load_and_process_data(self, file_path):
        """Load and process input data"""
        with open(file_path, 'r') as file:
            data = [json.loads(line) for line in file]
        return pd.DataFrame(data)

    def classify(self, data_path):
        """Main classification function"""
        # load_data
        data = self.load_and_process_data(data_path)

        # k2p
        data = self.predict_and_evaluate('k2p.pkl', data, ['seq_vector10', 'backseq_vector10'], 'p', 'pre_p')

        # data_group
        partitioned_data = {p: df for p, df in data.groupby('pre_p')}
        basidiomycota_data = partitioned_data.get('Basidiomycota', pd.DataFrame())
        ascomycota_data = partitioned_data.get('Ascomycota', pd.DataFrame())
        other_data = pd.concat([df for p, df in partitioned_data.items() if p not in ['Basidiomycota', 'Ascomycota']])

        # other_p
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

        # res_concat
        result = pd.concat([other_data, ascomycota_data, basidiomycota_data], axis=0).sort_index()
        result.drop(columns=['seq_vector7', 'backseq_vector7', 'seq_vector10', 'backseq_vector10'], inplace=True)
        result = result[
            ['seq', 'k', 'p', 'c', 'o', 'f', 'g', 's', 'pre_p', 'pre_c', 'pre_o', 'pre_f', 'pre_g', 'pre_s']]

        for col in ['p', 'c', 'o', 'f', 'g', 's']:
            accuracy = (result[col] == result[f'pre_{col}']).mean()
            logging.info(f"Overall accuracy for {col}: {accuracy:.4f}")

        return result

    def get_metrics(self, df, prefixs):
        """Calculate the indicators"""
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

    def get_HA(self, df, prefixs):
        """HA (Hierarchical Accuracy)"""
        HA = 0
        for i, row in df.iterrows():
            if all(row[pre] == row[f'pre_{pre}'] for pre in prefixs):
                HA += 1
        return HA / df.shape[0]


def main():
    # Set command-line parameter parsing
    parser = argparse.ArgumentParser(description="Hierarchical Classification Tool")
    parser.add_argument('--classify', type=str, help="Path to the input data for classification")
    parser.add_argument('--metrics', type=str, help="Path to the classification result for metrics calculation")
    parser.add_argument('--output', type=str, default="metrics.csv", help="Path to save the metrics output")
    args = parser.parse_args()

    # Initialize the classifier
    hc = HierarchicalClassification()

    if args.classify:
        # Perform classification
        res = hc.classify(args.classify)
        res.to_csv("classification_result.csv", index=False)
        logging.info("Classification completed. Results saved to 'classification_result.csv'.")

    if args.metrics:
        # Calculate the metrics
        df = pd.read_csv(args.metrics)
        prefixs = ['p', 'c', 'o', 'f', 'g', 's']
        metrics = hc.get_metrics(df, prefixs)
        metrics.to_csv(args.output, index=False)
        logging.info(f"Metrics calculation completed. Results saved to '{args.output}'.")


if __name__ == '__main__':
    main()