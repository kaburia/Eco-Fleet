import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import osmnx as ox
from osmnx import nearest_nodes
import networkx as nx
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim

# Initialize Streamlit app
st.set_page_config(page_title="Ecofleet EV Optimization", page_icon="🚕", layout="wide")
st.title("Ecofleet EV Range Optimization")
st.markdown("Plan your trips efficiently with EV range predictions and route optimization.")

# Session state initialization
if 'trip_results' not in st.session_state:
    st.session_state['trip_results'] = None

if 'combined_map' not in st.session_state:
    st.session_state['combined_map'] = None

# Load Charging Stations Data
@st.cache_data
def load_charging_data():
    df = pd.read_csv("Charging_stationKE1.csv")
    return df

try:
    charging_stations_df = load_charging_data()
    st.markdown("_Charging station data loaded successfully._")
except Exception as e:
    st.error(f"Error loading charging stations data: {e}")

# Create a base Folium map with charging station markers
def create_base_map(charging_df, center_coords):
    base_map = folium.Map(location=center_coords, zoom_start=13)
    # Add charging station markers
    for idx, row in charging_df.iterrows():
        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            popup=f"Charging Station {idx}",
            icon=folium.Icon(color='blue', icon='bolt', prefix='fa')
        ).add_to(base_map)
    return base_map

# Generate Synthetic Training Data and Train Model
np.random.seed(42)
n_samples = 1000
soc = np.random.uniform(20, 100, n_samples)
voltage = soc * 0.36 + np.random.uniform(0.9, 1.1, n_samples)
current = np.random.uniform(5, 20, n_samples)
speed = np.random.uniform(10, 40, n_samples)
temperature = np.random.uniform(0, 40, n_samples)
gradient = np.random.uniform(0, 15, n_samples)
range_km = (soc * 0.5) - (current * 1.2) - (speed * 0.4) - (gradient * 0.3) + (40 - temperature) * 0.2
range_km = np.clip(range_km, 0, None)

data = pd.DataFrame({
    'soc': soc,
    'voltage': voltage,
    'current': current,
    'speed': speed,
    'temperature': temperature,
    'gradient': gradient,
    'range_km': range_km
})

X = data[['soc', 'voltage', 'current', 'speed', 'temperature', 'gradient']]
y = data['range_km']
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Prediction Function
def predict_range(soc, voltage, current, speed, temperature, gradient):
    input_data = pd.DataFrame({
        'soc': [soc],
        'voltage': [voltage],
        'current': [current],
        'speed': [speed],
        'temperature': [temperature],
        'gradient': [gradient]
    })
    predicted_range = model.predict(input_data)[0]
    return predicted_range

# Calculate Shortest Route
@st.cache_data
def calculate_shortest_route(start_coords, end_coords, search_radius):
    try:
        G = ox.graph_from_point(start_coords, dist=search_radius, network_type='drive')
        if len(G.nodes) == 0 or len(G.edges) == 0:
            st.warning("No roads found within the search radius. Try increasing the radius.")
            return None, None, None
        start_node = ox.nearest_nodes(G, start_coords[1], start_coords[0])
        end_node = ox.nearest_nodes(G, end_coords[1], end_coords[0])
        route = nx.shortest_path(G, start_node, end_node, weight='length')
        route_length = nx.shortest_path_length(G, start_node, end_node, weight='length')
        return route, route_length, G
    except Exception as e:
        st.error(f"An error occurred during route calculation: {e}")
        return None, None, None

# Calculate Trip Range and Get Route Data
def calculate_trip_range(start_coords, end_coords, soc, temperature, search_radius):
    route, route_length, G = calculate_shortest_route(start_coords, end_coords, search_radius)
    if route is None or route_length is None:
        return None, None, None, None, None

    distance_km = route_length / 1000
    avg_speed = 30
    # For demo purposes, random values for gradient and current are used
    gradient = np.random.uniform(0, 10)
    current = np.random.uniform(5, 10)
    predicted_range = predict_range(soc, soc * 0.36, current, avg_speed, temperature, gradient)
    can_complete_trip = predicted_range >= distance_km

    return predicted_range, distance_km, can_complete_trip, route, G

# Sidebar Input
def geocode_place(place_name):
    common_places = {
        "Nairobi": (-1.286389, 36.817223),
        "Westlands": (-1.268290, 36.811060),
        "Kilimani": (-1.292066, 36.783377),
        "Thika": (-1.032214, 37.069328),
        "Karen": (-1.310097, 36.720037),
        "Ruiru": (-1.138411, 36.962763),
    }
    if place_name in common_places:
        return common_places[place_name]

    geolocator = Nominatim(user_agent="ev_optimization_app", timeout=10)
    try:
        location = geolocator.geocode(place_name)
        if location:
            return location.latitude, location.longitude
        else:
            st.warning(f"Could not find coordinates for: {place_name}")
            return None, None
    except Exception as e:
        st.error(f"An error occurred during geocoding: {e}")
        return None, None

st.sidebar.header("Trip Details")
start_location = st.sidebar.text_input("Enter start location:", "Nairobi")
end_location = st.sidebar.text_input("Enter destination:", "Westlands")
soc = st.sidebar.slider("State of Charge (%):", 20, 100, 80)
temperature = st.sidebar.slider("Temperature (°C):", 0, 40, 25)
search_radius = st.sidebar.slider("Search Radius (meters):", 500, 5000, 1000)

if st.sidebar.button("Calculate Trip"):
    start_coords = geocode_place(start_location)
    end_coords = geocode_place(end_location)
    if start_coords and end_coords:
        predicted_range, distance_km, can_complete_trip, route, G = calculate_trip_range(
            start_coords, end_coords, soc, temperature, search_radius
        )
        if predicted_range is not None:
            st.session_state['trip_results'] = {
                'predicted_range': predicted_range,
                'distance_km': distance_km,
                'can_complete_trip': can_complete_trip
            }
            # Create a combined map using the start_coords as the center
            combined_map = create_base_map(charging_stations_df, center_coords=start_coords)
            if route and G:
                # Extract route coordinates
                route_coords = [(G.nodes[node]['y'], G.nodes[node]['x']) for node in route]
                # Add the route polyline
                folium.PolyLine(route_coords, color="blue", weight=5, opacity=0.8).add_to(combined_map)
                # Mark start and end points
                folium.Marker(route_coords[0], popup="Start", icon=folium.Icon(color="green")).add_to(combined_map)
                folium.Marker(route_coords[-1], popup="End", icon=folium.Icon(color="red")).add_to(combined_map)
            st.session_state['combined_map'] = combined_map

# If no trip has been calculated, show a default base map centered on the mean location of charging stations
if st.session_state['combined_map'] is None:
    default_center = [charging_stations_df['Latitude'].mean(), charging_stations_df['Longitude'].mean()]
    st.session_state['combined_map'] = create_base_map(charging_stations_df, center_coords=default_center)

# Display Trip Results
if st.session_state['trip_results']:
    results = st.session_state['trip_results']
    st.subheader("Trip Results")
    st.write(f"Predicted Range: {results['predicted_range']:.2f} km")
    st.write(f"Trip Distance: {results['distance_km']:.2f} km")
    st.write("Can Complete Trip: " + ("Yes" if results['can_complete_trip'] else "No"))

# Display the Combined Map with both charging stations and (if available) the trip route
st.subheader("Map")
st_folium(st.session_state['combined_map'], width=700, height=500)
