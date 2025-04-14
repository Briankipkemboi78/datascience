# Deforestation Risk Prediction System
# Libraries to install:
# pip install geopandas pandas numpy matplotlib scikit-learn xgboost folium

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import folium
from folium.plugins import HeatMap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import xgboost as xgb
import warnings
from datetime import datetime
import json

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Geometry is in a geographic CRS")

class DeforestationRiskPredictor:
    def __init__(self, region_geojson_path, historical_deforestation_csv_path):
        """
        Initialize the deforestation risk prediction system.
        
        Parameters:
        -----------
        region_geojson_path : str
            Path to GeoJSON file containing region boundaries
        historical_deforestation_csv_path : str
            Path to CSV file containing historical deforestation incidents
        """
        self.region_gdf = gpd.read_file(region_geojson_path)
        self.historical_df = pd.read_csv(historical_deforestation_csv_path)
        self.model = None
        self.feature_importance = None
        self.risk_map = None
        
    def preprocess_data(self):
        """Preprocess and merge all data sources"""
        print("Preprocessing data...")
        
        # Convert historical data to GeoDataFrame
        self.historical_gdf = gpd.GeoDataFrame(
            self.historical_df,
            geometry=gpd.points_from_xy(self.historical_df['longitude'], self.historical_df['latitude']),
            crs="EPSG:4326"
        )
        
        # Spatial join to associate deforestation events with regions
        self.data = gpd.sjoin(self.region_gdf, self.historical_gdf, how="left", predicate='contains')
        
        # Create target variable: regions with deforestation incidents = 1, others = 0
        self.data['deforestation'] = self.data['index_right'].notna().astype(int)
        self.data = self.data.drop(columns=['index_right'])
        
        # Generate simulated environmental data instead of using Earth Engine
        self.add_simulated_features()
        
        return self.data
    
    def add_simulated_features(self):
        """Add simulated environmental features (to replace Earth Engine)"""
        print("Adding simulated environmental features...")
        
        # Calculate centroids for regions (with CRS warning suppressed by import-level warning filter)
        self.data['centroid'] = self.data.geometry.centroid
        self.data['centroid_lon'] = self.data.centroid.x
        self.data['centroid_lat'] = self.data.centroid.y
        
        # Generate simulated environmental features
        num_regions = len(self.data)
        
        # NDVI: typically ranges from -1 to 1, higher in forested areas
        self.data['ndvi'] = np.random.uniform(0.2, 0.9, size=num_regions)
        
        # Elevation: in meters
        self.data['elevation'] = np.random.uniform(100, 2000, size=num_regions)
        
        # Distance to roads: in meters
        self.data['distance_to_roads'] = np.random.uniform(100, 10000, size=num_regions)
        
        # Land cover: categorical (1=forest, 2=agriculture, 3=urban, 4=water, 5=other)
        self.data['landcover'] = np.random.choice([1, 2, 3, 4, 5], size=num_regions, p=[0.5, 0.3, 0.1, 0.05, 0.05])
        
        # Historical forest loss: binary (0=no loss, 1=loss)
        self.data['forest_loss_history'] = np.random.choice([0, 1], size=num_regions, p=[0.7, 0.3])
        
        # Rainfall: annual average in mm
        self.data['rainfall_mm'] = np.random.uniform(800, 3000, size=num_regions)
        
        # Temperature: average annual in Celsius
        self.data['temperature_c'] = np.random.uniform(15, 30, size=num_regions)
        
        # Slope: in degrees
        self.data['slope_deg'] = np.random.uniform(0, 45, size=num_regions)
        
        # Add correlations between features and deforestation
        # Regions with deforestation tend to have:
        # - Lower NDVI (vegetation removed)
        # - Closer to roads (accessibility)
        # - More historical forest loss
        deforestation_mask = self.data['deforestation'] == 1
        if deforestation_mask.any():
            # Adjust NDVI to be lower in deforested areas
            self.data.loc[deforestation_mask, 'ndvi'] = np.random.uniform(0.1, 0.5, size=deforestation_mask.sum())
            
            # Make deforested areas closer to roads
            self.data.loc[deforestation_mask, 'distance_to_roads'] = np.random.uniform(100, 5000, size=deforestation_mask.sum())
            
            # Increase likelihood of historical forest loss
            self.data.loc[deforestation_mask, 'forest_loss_history'] = np.random.choice([0, 1], size=deforestation_mask.sum(), p=[0.3, 0.7])
            
            # Make deforested areas more likely to be in agricultural zones
            self.data.loc[deforestation_mask, 'landcover'] = np.random.choice([1, 2, 3, 4, 5], size=deforestation_mask.sum(), p=[0.3, 0.5, 0.1, 0.05, 0.05])
        
    def prepare_features(self):
        """Prepare features for modeling"""
        print("Preparing features...")
        
        # Define feature columns - using our simulated data
        feature_cols = [
            'ndvi', 'elevation', 'distance_to_roads', 'landcover', 
            'forest_loss_history', 'rainfall_mm', 'temperature_c', 'slope_deg'
        ]
        
        # Make sure we have area and population density
        if 'area_km2' not in self.data.columns:
            # Calculate area in km² (approximate for demo purposes)
            self.data['area_km2'] = self.data.geometry.area / 10000  # Convert to km²
            feature_cols.append('area_km2')
            
        if 'population_density' not in self.data.columns:
            # Simulate population density with random values
            self.data['population_density'] = np.random.randint(1, 500, size=len(self.data))
            feature_cols.append('population_density')
        
        # Handle missing values
        for col in feature_cols:
            if col in self.data.columns:
                self.data[col] = self.data[col].fillna(self.data[col].median())
            else:
                # If column doesn't exist, create it with dummy values
                self.data[col] = np.random.random(len(self.data))
        
        # Keep only necessary columns
        self.features = self.data[feature_cols]
        self.target = self.data['deforestation']
        
        # Scale features
        scaler = StandardScaler()
        self.features_scaled = scaler.fit_transform(self.features)
        
        return self.features_scaled, self.target
    
    def train_model(self):
        """Train XGBoost model on the prepared data"""
        print("Training model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.features_scaled, self.target, test_size=0.3, random_state=42
        )
        
        # Define and train XGBoost model
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            min_child_weight=1,
            gamma=0,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        print("\nModel Evaluation:")
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))
        
        # Get feature importance
        self.feature_importance = pd.DataFrame({
            'Feature': self.features.columns,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("\nFeature Importance:")
        print(self.feature_importance)
        
        # Predict risk for all regions
        self.data['risk_score'] = self.model.predict_proba(self.features_scaled)[:, 1]
        
        # Handle edge case where all probabilities might be the same
        try:
            self.data['risk_category'] = pd.qcut(
                self.data['risk_score'], 
                q=[0, 0.25, 0.5, 0.75, 1.0], 
                labels=['Low', 'Medium', 'High', 'Very High']
            )
        except ValueError:
            # If qcut fails (e.g., all values are identical), use cut instead
            min_val = self.data['risk_score'].min()
            max_val = self.data['risk_score'].max()
            if min_val == max_val:
                self.data['risk_category'] = 'Medium'  # Assign all to medium if all scores are the same
            else:
                bins = [min_val, min_val + (max_val-min_val)/4, min_val + 2*(max_val-min_val)/4, 
                        min_val + 3*(max_val-min_val)/4, max_val]
                self.data['risk_category'] = pd.cut(
                    self.data['risk_score'],
                    bins=bins,
                    labels=['Low', 'Medium', 'High', 'Very High'],
                    include_lowest=True
                )
        
        return self.model
    
    def identify_risk_factors(self):
        """Identify top risk factors for each high-risk region"""
        print("Identifying risk factors...")
    
    # Filter high-risk regions based on the risk category
        high_risk_regions = self.data[self.data['risk_category'].isin(['High', 'Very High'])]
    
    # Check if there are any high-risk regions to process
        if high_risk_regions.empty:
            print("\nNo high-risk regions identified.")
            return pd.DataFrame()  # Return empty DataFrame
    
    # List to store results
        risk_factors = []
    
    # Feature names from the features dataframe
        feature_names = self.features.columns
    
    # Iterate over each high-risk region
        for idx, row in high_risk_regions.iterrows():
        # Get feature values for this region
            region_features = self.features.loc[idx].values
        
        # Ensure dimensions match
            if len(region_features) != len(self.model.feature_importances_):
                print(f"Warning: Dimension mismatch for region {row.get('name', f'Region {idx}')}")
                continue
        
        # Calculate feature contributions (feature_value * importance)
            feature_contributions = np.zeros(len(feature_names))
        
            for i, (feature_val, importance) in enumerate(zip(region_features, self.model.feature_importances_)):
            # Check if feature_val is scalar
                if isinstance(feature_val, (np.ndarray, list)):
                    print(f"Warning: feature_val is not scalar for region {row.get('name', f'Region {idx}')} at feature {feature_names[i]}")
                    continue  # Skip this feature if it's not scalar
            
            # Check if importance is scalar
                if isinstance(importance, (np.ndarray, list)):
                    print(f"Warning: importance is not scalar for feature {feature_names[i]}")
                    continue  # Skip if importance is not scalar
            
            # Assign the product of feature_val and importance
                feature_contributions[i] = feature_val * importance
        
        # Get top 3 contributing features
            top_indices = np.argsort(feature_contributions)[-3:][::-1]
            top_features = [feature_names[i] for i in top_indices]
        
        # Append result for this region
            risk_factors.append({
                'region_name': row.get('name', f'Region {idx}'),
                'risk_score': row['risk_score'],
                'top_factors': top_features
            })
    
    # Convert the risk factors list to a DataFrame
        self.risk_factors = pd.DataFrame(risk_factors)
    
    # Print a sample of the results if any
        if len(self.risk_factors) > 0:
            print("\nRisk Factors for High-Risk Regions:")
            print(self.risk_factors.head())
        else:
            print("\nNo high-risk regions identified.")
    
        return self.risk_factors
    
    def create_risk_map(self, output_html_path="deforestation_risk_map.html"):
        """Create a folium map with risk visualization"""
        print("Creating risk map...")
        
        # Create a GeoDataFrame with risk categories
        risk_gdf = self.data.copy()
        
        # Create color mapping
        risk_colors = {
            'Low': 'green',
            'Medium': 'yellow',
            'High': 'orange',
            'Very High': 'red'
        }
        
        # Create a JSON serializable GeoDataFrame
        # First, convert to GeoJSON
        risk_json = json.loads(risk_gdf.to_json())
        
        # Create base map
        center_lat = risk_gdf.geometry.centroid.y.mean()
        center_lon = risk_gdf.geometry.centroid.x.mean()
        m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
        
        # Function to get the risk category for a feature
        def get_style(feature):
            feature_id = feature['id']
            idx = int(feature_id) if feature_id is not None and str(feature_id).isdigit() else 0
            if idx < len(risk_gdf):
                risk_category = risk_gdf.iloc[idx]['risk_category']
                color = risk_colors.get(risk_category, 'gray')
            else:
                color = 'gray'
            return {
                'fillColor': color,
                'color': 'black',
                'weight': 1,
                'fillOpacity': 0.7
            }
        
        # Function to create tooltip content
        def get_tooltip(feature):
            feature_id = feature['id']
            idx = int(feature_id) if feature_id is not None and str(feature_id).isdigit() else 0
            if idx < len(risk_gdf):
                row = risk_gdf.iloc[idx]
                return f"Region: {row.get('name', f'Region {idx}')}<br>Risk Score: {row['risk_score']:.3f}<br>Risk Category: {row['risk_category']}"
            else:
                return "Region data not available"
        
        # Add GeoJSON layer using the processed GeoJSON data
        folium.GeoJson(
            risk_json,
            name='Deforestation Risk',
            style_function=get_style,
            tooltip=folium.GeoJsonTooltip(
                fields=['id'],
                aliases=['Region:'],
                style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;"),
                sticky=True,
            )
        ).add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Save map
        m.save(output_html_path)
        print(f"Risk map saved to {output_html_path}")
        
        self.risk_map = m
        return m
    
    def create_dashboard(self, output_dir="dashboard_output"):
        """Create a comprehensive dashboard with risk map and analysis"""
        print("Creating dashboard...")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate risk map
        self.create_risk_map(output_html_path=os.path.join(output_dir, "risk_map.html"))
        
        # Generate feature importance plot
        plt.figure(figsize=(10, 6))
        self.feature_importance.sort_values('Importance').plot(kind='barh', x='Feature', y='Importance')
        plt.title('Feature Importance for Deforestation Risk')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "feature_importance.png"))
        
        # Generate risk distribution plot
        plt.figure(figsize=(10, 6))
        self.data['risk_score'].hist(bins=20)
        plt.title('Distribution of Deforestation Risk Scores')
        plt.xlabel('Risk Score')
        plt.ylabel('Number of Regions')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "risk_distribution.png"))
        
        # Generate HTML dashboard
        dashboard_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Deforestation Risk Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .header {{ background-color: #4CAF50; color: white; padding: 20px; text-align: center; }}
                .row {{ display: flex; flex-wrap: wrap; margin: 0 -15px; }}
                .col {{ flex: 50%; padding: 15px; }}
                .card {{ background-color: white; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); 
                       margin-bottom: 20px; padding: 20px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                iframe {{ border: none; width: 100%; height: 500px; }}
                img {{ max-width: 100%; height: auto; }}
                .risk-very-high {{ background-color: #ffcccc; }}
                .risk-high {{ background-color: #ffeecc; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Deforestation Risk Analysis Dashboard</h1>
                    <p>Analysis date: {datetime.now().strftime('%Y-%m-%d')}</p>
                </div>
                
                <div class="row">
                    <div class="col">
                        <div class="card">
                            <h2>Risk Map</h2>
                            <iframe src="risk_map.html"></iframe>
                            <p>Interactive map showing deforestation risk levels across regions.</p>
                        </div>
                    </div>
                    
                    <div class="col">
                        <div class="card">
                            <h2>Risk Factor Analysis</h2>
                            <img src="feature_importance.png" alt="Feature Importance">
                            <p>The chart above shows the relative importance of each factor in predicting deforestation risk.</p>
                        </div>
                        
                        <div class="card">
                            <h2>Risk Distribution</h2>
                            <img src="risk_distribution.png" alt="Risk Distribution">
                            <p>Distribution of risk scores across all analyzed regions.</p>
                        </div>
                    </div>
                </div>
                
                <div class="card">
                    <h2>High-Risk Regions</h2>
                    <table>
                        <tr>
                            <th>Region</th>
                            <th>Risk Score</th>
                            <th>Risk Category</th>
                            <th>Top Risk Factors</th>
                        </tr>
        """
        
        # Add high-risk regions to the table
        high_risk = self.data[self.data['risk_category'].isin(['High', 'Very High'])].sort_values('risk_score', ascending=False)
        for idx, row in high_risk.head(10).iterrows():
            region_name = row.get('name', f'Region {idx}')
            risk_score = f"{row['risk_score']:.3f}"
            risk_category = row['risk_category']
            
            # Get top factors for this region
            if hasattr(self, 'risk_factors'):
                region_factors = self.risk_factors[self.risk_factors['region_name'] == region_name]
                if not region_factors.empty:
                    top_factors = ", ".join(region_factors.iloc[0]['top_factors'])
                else:
                    top_factors = "N/A"
            else:
                top_factors = "N/A"
            
            css_class = "risk-very-high" if risk_category == "Very High" else "risk-high"
            dashboard_html += f"""
                <tr class="{css_class}">
                    <td>{region_name}</td>
                    <td>{risk_score}</td>
                    <td>{risk_category}</td>
                    <td>{top_factors}</td>
                </tr>
            """
        
        dashboard_html += """
                    </table>
                </div>
                
                <div class="card">
                    <h2>Recommendations</h2>
                    <ul>
                        <li>Increase monitoring frequency in Very High risk regions</li>
                        <li>Develop community-based forest protection programs in high-risk areas</li>
                        <li>Address key risk factors identified in the analysis</li>
                        <li>Implement early warning systems in High and Very High risk zones</li>
                        <li>Enhance law enforcement presence in areas near roads with high deforestation risk</li>
                    </ul>
                </div>
                
                <div class="header" style="background-color: #333; margin-top: 20px;">
                    <p>Deforestation Risk Prediction System | Generated on {date}</p>
                </div>
            </div>
        </body>
        </html>
        """.format(date=datetime.now().strftime('%Y-%m-%d'))
        
        # Save dashboard HTML
        with open(os.path.join(output_dir, "dashboard.html"), "w") as f:
            f.write(dashboard_html)
            
        print(f"Dashboard created in {output_dir}")
        return output_dir


# Sample usage
def generate_sample_data(output_dir="sample_data"):
    """Generate sample data for demonstration"""
    print("Generating sample data...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create sample region boundaries (GeoJSON)
    # Create a grid of polygons
    from shapely.geometry import Polygon
    
    polygons = []
    region_data = []
    
    for i in range(5):
        for j in range(5):
            # Create a polygon (square)
            min_x, min_y = -74.0 + j*0.1, 4.5 + i*0.1
            max_x, max_y = min_x + 0.1, min_y + 0.1
            
            polygon = Polygon([
                (min_x, min_y), (max_x, min_y),
                (max_x, max_y), (min_x, max_y)
            ])
            
            polygons.append(polygon)
            region_data.append({
                'id': i*5 + j,
                'name': f'Region {i*5 + j}',
                'area_km2': np.random.uniform(10, 100),
                'population_density': np.random.uniform(10, 500)
            })
    
    # Create GeoDataFrame with regions
    region_gdf = gpd.GeoDataFrame(region_data, geometry=polygons, crs="EPSG:4326")
    region_gdf.to_file(os.path.join(output_dir, "sample_regions.geojson"), driver="GeoJSON")
    
    # Create sample historical deforestation data (CSV)
    # Simulate some deforestation events within the regions
    deforestation_data = []
    
    # Add some random deforestation events
    for _ in range(30):
        region_idx = np.random.randint(0, len(polygons))
        region = polygons[region_idx]
        
        # Get a random point within the region (simple approximation)
        minx, miny, maxx, maxy = region.bounds
        lon = np.random.uniform(minx, maxx)
        lat = np.random.uniform(miny, maxy)
        
        deforestation_data.append({
            'latitude': lat,
            'longitude': lon,
            'date': f"2020-{np.random.randint(1, 13):02d}-{np.random.randint(1, 29):02d}",
            'area_hectares': np.random.uniform(1, 10),
            'cause': np.random.choice(['Agriculture', 'Logging', 'Infrastructure', 'Mining', 'Unknown'], p=[0.4, 0.3, 0.1, 0.1, 0.1])
        })
    
    deforestation_df = pd.DataFrame(deforestation_data)
    deforestation_df.to_csv(os.path.join(output_dir, "historical_deforestation.csv"), index=False)
    
    print(f"Sample data generated in {output_dir}")
    return os.path.join(output_dir, "sample_regions.geojson"), os.path.join(output_dir, "historical_deforestation.csv")


def main():
    # Generate sample data
    region_geojson_path, historical_deforestation_csv_path = generate_sample_data()
    
    # Initialize and run the deforestation risk prediction system
    risk_predictor = DeforestationRiskPredictor(
        region_geojson_path=region_geojson_path,
        historical_deforestation_csv_path=historical_deforestation_csv_path
    )
    
    # Process data and train model
    risk_predictor.preprocess_data()
    risk_predictor.prepare_features()
    risk_predictor.train_model()
    risk_predictor.identify_risk_factors()
    
    # Create dashboard
    dashboard_path = risk_predictor.create_dashboard()
    print(f"\nProcess complete! Dashboard available at: {dashboard_path}/dashboard.html")


if __name__ == "__main__":
    main()