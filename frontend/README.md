# Housing Price Prediction Frontend

A clean, minimal, and responsive web frontend for the California Housing Price Prediction API.

## Features

- **Responsive Design**: Works seamlessly on mobile, tablet, and desktop devices
- **Form Validation**: Client-side validation with real-time feedback
- **API Integration**: Connects to the FastAPI backend for predictions
- **Error Handling**: Comprehensive error handling with user-friendly messages
- **Loading States**: Visual feedback during API requests
- **Currency Formatting**: Professional display of predicted prices

## Usage

1. **Development**: Open `index.html` in your browser
2. **Production**: Serve the file using any web server

## Configuration

The frontend is configured to connect to the backend at `http://localhost:8000`. To change this:

1. Open `index.html`
2. Find the line: `this.apiUrl = 'http://localhost:8000/predict';`
3. Update the URL to match your backend deployment

## Input Fields

The form collects the following housing data:

- **Longitude**: Decimal number (-180 to 180)
- **Latitude**: Decimal number (-90 to 90)  
- **Median Age**: Housing age in years (0-100)
- **Total Rooms**: Number of rooms (minimum 1)
- **Total Bedrooms**: Number of bedrooms (minimum 1)
- **Population**: Area population (minimum 1)
- **Households**: Number of households (minimum 1)
- **Median Income**: Income in tens of thousands (e.g., 8.3252 = $83,252)
- **Ocean Proximity**: Dropdown with options (Near Bay, Near Ocean, <1H Ocean, Inland, Island)

## API Response

The frontend expects the following response format from the `/predict` endpoint:

```json
{
  "predictions": [2.408],
  "model_version": "20250909_130932",
  "metadata": {
    "train_metrics": {"rmse": 49401.57, "r2": 0.8239},
    "feature_names": ["longitude", "latitude", ...]
  }
}
```

## Browser Support

- Modern browsers (Chrome, Firefox, Safari, Edge)
- Mobile browsers (iOS Safari, Chrome Mobile)
- Responsive design works on all screen sizes