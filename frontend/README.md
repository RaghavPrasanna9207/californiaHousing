# Housing Price Prediction Frontend

A modern, responsive web frontend for the California Housing Price Prediction API with real-time validation, loading states, and an intuitive user experience.

## Features

- **🎨 Modern UI Design**: Clean purple gradient theme with professional styling
- **📱 Fully Responsive**: Optimized for desktop, tablet, and mobile devices
- **✅ Smart Validation**: Real-time input validation with contextual error messages
- **🔄 Loading States**: Visual feedback with spinners and form overlays during API requests
- **🔄 Reset Functionality**: One-click form reset with validation cleanup
- **🎯 Focus Management**: Intelligent focus handling and accessibility features
- **💰 Currency Formatting**: Professional display of predicted housing prices
- **🔗 API Integration**: Seamless connection to FastAPI backend
- **♿ Accessibility**: WCAG compliant with proper ARIA labels and keyboard navigation

## Quick Start

### Prerequisites
Ensure the backend API is running on `http://localhost:8000`. See the main README for backend setup instructions.

### Running the Frontend
1. **Local Development**: Simply open `frontend/index.html` in your web browser
2. **Web Server**: For production, serve the files using any web server (nginx, Apache, etc.)

## Configuration

### API Endpoint Configuration
The frontend connects to the backend at `http://localhost:8000/predict` by default. To change this:

1. Open `index.html`
2. Find the line in the JavaScript: `this.apiUrl = 'http://localhost:8000/predict';`
3. Update the URL to match your backend deployment

## User Interface

### Form Fields
The prediction form collects the following California housing data:

| Field | Description | Validation |
|-------|-------------|------------|
| **Longitude** | Geographic longitude coordinate | -180 to 180 |
| **Latitude** | Geographic latitude coordinate | -90 to 90 |
| **Median Age** | Housing median age in years | 0-100 years |
| **Average Rooms** | Average number of rooms per household | Minimum 1 |
| **Average Bedrooms** | Average number of bedrooms per household | Minimum 0.1 |
| **Population** | Area population count | Minimum 1 |
| **Average Occupancy** | Average occupancy per household | Minimum 0.1 |
| **Median Income** | Median income in tens of thousands | Minimum 0 (e.g., 8.3252 = $83,252) |

### Validation Features
- **Real-time validation**: Error messages appear only when fields are focused or invalid
- **Contextual help**: Validation messages provide specific guidance
- **Visual feedback**: Invalid fields are highlighted with red borders
- **Form state management**: Submit button disabled during validation errors

### Loading States
- **Button loading**: Submit button shows spinner during API requests
- **Form overlay**: Semi-transparent overlay prevents interaction during loading
- **Visual feedback**: Clear indication of request progress

### Results Display
- **Success state**: Green gradient card with formatted price prediction
- **Error state**: Red gradient card with error message details
- **Model metadata**: Displays model version and accuracy metrics when available
- **Currency formatting**: Professional USD formatting with thousands separators

## API Response

### Expected Response Format
The frontend expects this JSON structure from the `/predict` endpoint:

```json
{
  "predictions": [2.408],
  "model_version": "20250909_130932",
  "metadata": {
    "train_metrics": {"rmse": 49401.57, "r2": 0.8239},
    "feature_names": ["MedInc", "HouseAge", "AveRooms", ...]
  }
}
```

### Error Handling
The frontend gracefully handles various error scenarios:
- **Network errors**: Connection issues with the backend
- **Validation errors**: Invalid input data format
- **Server errors**: Backend processing failures
- **Timeout errors**: Long-running prediction requests

## Technical Details

### Architecture
- **Vanilla JavaScript**: No external dependencies for maximum compatibility
- **CSS Grid & Flexbox**: Modern layout techniques for responsive design
- **CSS Custom Properties**: Consistent theming and easy customization
- **Progressive Enhancement**: Works without JavaScript for basic functionality

### Browser Compatibility
- **Modern browsers**: Chrome 88+, Firefox 85+, Safari 14+, Edge 88+
- **Mobile browsers**: iOS Safari 14+, Chrome Mobile 88+
- **Responsive breakpoints**: 1200px (desktop), 900px (tablet), 600px (mobile)

### Performance Features
- **Optimized CSS**: Efficient selectors and minimal reflows
- **Smooth animations**: Hardware-accelerated transitions
- **Reduced motion support**: Respects user accessibility preferences
- **Fast loading**: Minimal external dependencies

## Customization

### Color Scheme
The design uses CSS custom properties for easy theming:

```css
:root {
  --color-primary-start: #4338ca;    /* Hero gradient start */
  --color-primary-end: #7c3aed;      /* Hero gradient end */
  --color-accent-start: #7c3aed;     /* Button gradient start */
  --color-accent-end: #c026d3;       /* Button gradient end */
  --color-success-start: #059669;    /* Success state */
  --color-error-start: #dc2626;      /* Error state */
}
```

### Typography
- **Font family**: Inter (Google Fonts) with system fallbacks
- **Font sizes**: Responsive scaling from 14px to 72px
- **Font weights**: 400 (regular), 500 (medium), 600 (semibold), 700 (bold), 800 (extrabold)

### Layout Customization
- **Container width**: Max 1100px with responsive padding
- **Form grid**: Two-column desktop layout, single-column mobile
- **Spacing system**: 8px base unit with consistent scaling

## Browser Support
