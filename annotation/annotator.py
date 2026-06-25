import pandas as pd
import os
import json
from openai import OpenAI
from config_utils import get_param

class CellAnnotator:
    def __init__(self, config):
        self.config = config
        self.api_key = get_param(config, "openai.api_key")
        self.model = get_param(config, "openai.model")
        self.is_thinking_model = get_param(config, "openai.is_thinking_model", False)
        
        # Set custom base_url if provided
        my_base_url = get_param(config, "openai.base_url", None)
        self.base_url = my_base_url if my_base_url else None
        
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.tissue = get_param(config, "annotation.tissue")

    def get_annotation(self, prompt):
        """
        Get annotation using OpenAI API
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a professional bioinformatician skilled in interpreting gene markers and cell types."},
                    {"role": "user", "content": prompt},
                ],
                # temperature=0.6,
                stream=False,
                max_tokens=100
            )
            content = response.choices[0].message.content.strip()
            print(content)
            
            if self.is_thinking_model:
                # Try to extract non-thinking content
                if "thinking:" in content.lower() and "\n" in content:
                    lines = content.split("\n")
                    filtered_lines = []
                    in_thinking_section = False
                    
                    for line in lines:
                        if "thinking:" in line.lower():
                            in_thinking_section = True
                            continue
                        if not in_thinking_section or line.strip() == "":
                            filtered_lines.append(line)
                    
                    content = "\n".join(filtered_lines).strip()
            
            return content
        except Exception as e:
            print(f"API call error: {e}")
            return None

    def read_markers_file(self, file_path):
        """
        Read markers_modality1.csv file and return DataFrame
        """
        try:
            df = pd.read_csv(file_path)
            return df
        except Exception as e:
            print(f"File reading error: {e}")
            return None

    def process_markers(self, df, top_n_genes=10):
        """
        Process markers data, extract genes for each column and get annotation
        """
        results = {}
        
        # Iterate through each column
        for column in df.columns:
            # Extract top_n_genes for this column (if available)
            genes = df[column].dropna().tolist()[:top_n_genes]
            
            # Remove b' prefix and ' suffix from gene names if present
            cleaned_genes = []
            for gene in genes:
                if isinstance(gene, str):
                    if gene.startswith("b'") and gene.endswith("'"):
                        gene = gene[2:-1]
                    cleaned_genes.append(gene)
                else:
                    cleaned_genes.append(str(gene))
            
            # Construct prompt
            prompt = f'''Given a list of highly expressed genes: {', '.join(cleaned_genes)} (note: genes listed earlier have higher relative weight), identify the most likely cell type (must belong to {self.tissue} tissue; the state can be used to describe cells, such as a certain type of rapidly proliferating cells or cancerous certain type cells, etc.) associated with these genes. Determine which specific genes from the provided list support this cell type identification. If the most probable cell type does not belong to {self.tissue} tissue, respond strictly with 'unknown, []'.
Output Requirements:
    Provide only a single full cell type name (no abbreviations) and a list of supporting genes in the exact format:
    cell_type_name, [gene1, gene2, ...]
    or
    unknown, []
    No explanations, notes, or additional text.
'''
            
            # Get annotation
            annotation = self.get_annotation(prompt)
            # annotation = "cell_type_name_test, [gene1, gene2]"

            if annotation:
                try:
                    cell_type, supporting_genes_str = annotation.split(", [", 1)
                    cell_type = cell_type.strip()
                    
                    # Process supporting genes part
                    supporting_genes = []
                    if supporting_genes_str != "[]":
                        supporting_genes = supporting_genes_str.strip("[]").split(", ")
                        supporting_genes = ", ".join(supporting_genes)
                    
                    # Store results  
                    results[column] = {
                        "genes": cleaned_genes,
                        "annotation": cell_type,
                        "supporting_genes": supporting_genes,
                    }
                except ValueError:
                    print(f"Error parsing annotation response for {column}: {annotation}")
                    results[column] = {
                        "genes": cleaned_genes,
                        "annotation": "unknown",
                        "supporting_genes": "",
                    }
            else:
                 results[column] = {
                        "genes": cleaned_genes,
                        "annotation": "unknown",
                        "supporting_genes": "",
                    }

        return results

    def add_labels_to_clusters(self, cluster_pred_file, cluster_labels_df):
        """
        Read cluster_pred.csv file and add corresponding cell type labels
        
        Args:
            cluster_pred_file: Path to cluster prediction file
            cluster_labels_df: DataFrame containing cluster label information, one row per category
        """
        # Read cluster prediction file
        try:
            cluster_pred = pd.read_csv(cluster_pred_file)
        except Exception as e:
            print(f"Error reading cluster prediction file: {e}")
            return None
        
        # Create cluster_id to label mapping
        cluster_to_label = {}
        supporting_genes = {}
        for idx, row in cluster_labels_df.iterrows():
            # Unify cluster ID format, remove possible "Cluster_" prefix
            cluster_id = str(row['Cluster']).replace("Cluster_", "")
            cluster_to_label[cluster_id] = row['Annotation']
            supporting_genes[cluster_id] = row['supporting_genes']
        
        # Find label for each cell
        def get_label(cluster):
            # Unify cluster ID format
            cluster_str = str(cluster).replace("Cluster_", "")
            return cluster_to_label.get(cluster_str, "unknown")

        def get_supporting_genes(cluster):
            # Unify cluster ID format
            cluster_str = str(cluster).replace("Cluster_", "")
            return supporting_genes.get(cluster_str, [])
        
        # Add new columns
        cluster_pred['Annotation'] = cluster_pred['cluster'].apply(get_label)
        cluster_pred['supporting_genes'] = cluster_pred['cluster'].apply(get_supporting_genes)
            
        # Print matching status
        print("\nCluster mapping:")
        for cluster_id, label in cluster_to_label.items():
            print(f"Cluster {cluster_id} -> {label}")
        
        # Print unmapped categories
        unique_clusters = set(cluster_pred['cluster'].astype(str).str.replace("Cluster_", ""))
        mapped_clusters = set(cluster_to_label.keys())
        unmapped = unique_clusters - mapped_clusters
        if unmapped:
            print("\nWarning: Following clusters have no labels:")
            for cluster in sorted(unmapped):
                print(f"Cluster {cluster}")
        
        return cluster_pred

    def run_annotation_pipeline(self):
        """
        Run the full annotation pipeline
        """
        output_path = get_param(self.config, "data.output_path")
        input_file_name = get_param(self.config, "data.input_file_name")
        top_n_genes = get_param(self.config, "annotation.top_n_genes", 20)
        
        markers_file_path = os.path.join(output_path, input_file_name, "markers_modality1.csv")
        print(f"Reading file: {markers_file_path}")
        
        markers_df = self.read_markers_file(markers_file_path)
        
        if markers_df is not None:
            results = self.process_markers(markers_df, top_n_genes)
            
            # Save results to JSON
            results_file_path = os.path.join(output_path, input_file_name, "cell_annotations.json")
            with open(results_file_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)

            # Save results to CSV
            csv_results = []
            for cluster, data in results.items():
                csv_results.append([cluster, data["annotation"], data["supporting_genes"]])
            
            csv_file_path = os.path.join(output_path, input_file_name, "cell_annotations.csv")
            csv_df = pd.DataFrame(csv_results, columns=["Cluster", "Annotation", "supporting_genes"])
            csv_df.to_csv(csv_file_path, index=False)

            # Add labels to cluster_pred.csv
            cluster_pred_file = f'{output_path}/{input_file_name}/cluster_pred.csv'
            if os.path.exists(cluster_pred_file):
                cluster_pred_with_labels = self.add_labels_to_clusters(
                    cluster_pred_file,
                    csv_df
                )
                if cluster_pred_with_labels is not None:
                    output_file = f'{output_path}/{input_file_name}/cluster_pred_with_labels.csv'
                    cluster_pred_with_labels.to_csv(output_file, index=False)
                    print(f"Annotation results saved to: {results_file_path} and {csv_file_path}")
                    print(f"Cluster predictions with labels saved to: {output_file}")
            else:
                 print(f"Annotation results saved to: {results_file_path} and {csv_file_path}")
                 print(f"Warning: {cluster_pred_file} not found, skipping cluster label mapping.")
        else:
            print("Unable to process markers file")
