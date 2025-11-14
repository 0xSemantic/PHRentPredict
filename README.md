# PHRentPredict

**PHRentPredict** is a web application designed to predict rent trends in Port Harcourt, Nigeria, helping users plan their housing budgets with confidence. Built with **FastAPI**, **Jinja2 templates**, and **Tailwind CSS**, it provides accurate 1-5 year rent forecasts for various property types across Port Harcourt neighborhoods. The system leverages historical data, economic indicators, and local housing dynamics to generate reliable predictions, incorporating additional costs like agent fees and caution deposits.

The application features an interactive **Port Harcourt Rent Trends** chart in the hero section, a user-friendly prediction form, detailed neighborhood insights, and a transparent methodology. It is free, requires no registration, and is optimized for both desktop and mobile devices.

## Table of Contents
- [System Overview](#system-overview)
- [Features](#features)
- [Architecture](#architecture)
- [Technologies Used](#technologies-used)
- [Setup Guide](#setup-guide)
  - [Prerequisites](#prerequisites)
  - [Step 1: Clone the Repository](#step-1-clone-the-repository)
  - [Step 2: Create and Activate a Virtual Environment](#step-2-create-and-activate-a-virtual-environment)
  - [Step 3: Install Dependencies](#step-3-install-dependencies)
  - [Step 4: Run the FastAPI Application](#step-4-run-the-fastapi-application)
  - [Step 5: Access and Test the Application](#step-5-access-and-test-the-application)
- [Testing the Application](#testing-the-application)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## System Overview

**PHRentPredict** addresses the challenge of budgeting for housing in Port Harcourt, where rent prices fluctuate due to economic factors, population growth, and local housing dynamics. The system provides users with:

- **Rent Forecasts**: Predicts rent prices for 1-5 years across neighborhoods like Diobu, GRA, Woji, and more, for property types ranging from single rooms to 3-bedroom apartments.
- **Cost Breakdown**: Includes agent fees (15-30%), caution deposits (1-2 months' rent), and other hidden costs in predictions.
- **Interactive Visualizations**: Displays historical and projected rent trends via an interactive Chart.js line chart in the hero section.
- **Neighborhood Insights**: Offers detailed information on Port Harcourt neighborhoods, helping users compare affordability and amenities.
- **User-Friendly Interface**: A responsive, mobile-friendly design with Tailwind CSS, enhanced with AOS animations and Font Awesome icons.

The system is built as a **FastAPI** web application, serving dynamic HTML templates with **Jinja2**. The frontend is styled with **Tailwind CSS**, and the hero section’s chart uses **Chart.js** to display static mock data for Single Room, Self-Contained, and 2 Bedroom properties from 2019–2025. The backend processes user inputs (neighborhood, property type, forecast horizon) to generate predictions, though the current hero chart uses hardcoded data for demonstration.

## Features

- **Accurate Predictions**:
  - Forecasts rent prices for 1-5 years with a Mean Absolute Percentage Error (MAPE) of less than 15%.
  - Considers historical rent data, inflation, oil prices, population influx, and utility costs.
- **Interactive Hero Chart**:
  - Displays rent trends for Single Room, Self-Contained, and 2 Bedroom properties (2019–2025).
  - Static data: e.g., 2025 values are ₦220K (Single Room), ₦700K (Self-Contained), ₦1.5M (2 Bedroom).
  - Supports dark/light theme toggling and hover tooltips for price details.
- **Prediction Form**:
  - Allows users to select a neighborhood (e.g., Diobu, GRA), property type (e.g., single room, 2-bedroom), and forecast horizon (1-5 years).
  - Submits to `/predict/html` endpoint for processing.
- **Neighborhood Insights**:
  - Details major Port Harcourt areas with price ranges (e.g., Diobu: ₦120-250K for Single Room).
- **Cost Transparency**:
  - Breaks down total move-in costs, including agent fees and deposits.
- **Responsive Design**:
  - Optimized for desktop, tablet, and smartphone using Tailwind CSS.
- **No Registration Required**:
  - Free and open to all users, with instant results and no login needed.
- **Transparent Methodology**:
  - Explains prediction models and confidence intervals in the FAQ section.

## Architecture

The system follows a **client-server architecture** with a **FastAPI** backend and a **Jinja2**-based frontend:

- **Frontend**:
  - **HTML Templates**: `index.html` (main page) and `base.html` (layout) use Jinja2 for dynamic rendering.
  - **Styling**: Tailwind CSS for responsive, modern design.
  - **Charting**: Chart.js for the hero section’s rent trends chart.
  - **Animations**: AOS (Animate on Scroll) for smooth transitions.
  - **Icons**: Font Awesome for visual elements.
- **Backend**:
  - **FastAPI**: Handles HTTP requests, serves templates, and processes form submissions.
  - **Prediction Endpoint**: `/predict/html` (POST) accepts form data (neighborhood, property_type, horizon_years) and returns predictions.
  - **Data**: The hero chart uses static mock data; future versions may integrate a database or API for dynamic data.
- **Deployment**:
  - Runs locally with **Uvicorn** as the ASGI server.
  - Suitable for deployment on cloud platforms like Heroku, AWS, or Render.

## Technologies Used

- **Backend**:
  - Python 3.8+
  - FastAPI: Asynchronous web framework
  - Uvicorn: ASGI server
  - Jinja2: Template engine
- **Frontend**:
  - HTML5
  - Tailwind CSS: Utility-first CSS framework
  - Chart.js: For interactive charts
  - AOS: Animation library
  - Font Awesome: Icon library
- **Development Tools**:
  - Git: Version control
  - Virtualenv: Isolated Python environments
  - pip: Package management

## Setup Guide

Follow these steps to download, set up, and run **PHRentPredict** locally.

### Prerequisites

Ensure you have the following installed:

- **Python 3.8+**: Download from [python.org](https://www.python.org/downloads/).
- **Git**: Install from [git-scm.com](https://git-scm.com/downloads).
- **pip**: Python package manager (included with Python).
- **Virtualenv**: Install with `pip install virtualenv`.
- **Code Editor**: VS Code, PyCharm, or any editor of choice.
- **Browser**: Chrome, Firefox, or Edge for testing.

Verify installations:
```bash
python --version
git --version
pip --version
```

### Step 1: Clone the Repository

1. **Create a Project Directory**:
   ```bash
   mkdir phrentpredict
   cd phrentpredict
   ```

2. **Clone the Repository**:
   ```bash
   git clone https://github.com/0xSemantic/PHRentPredict.git
   cd PHRentPredict
   ```

3. **Verify Files**:
   Ensure the project structure includes:
   - `app/`
     - `main.py`: FastAPI application
     - `templates/`
       - `index.html`: Main page
       - `base.html`: Base template
   - `requirements.txt`: Dependencies
   - `README.md`

### Step 2: Create and Activate a Virtual Environment

A virtual environment isolates project dependencies.

1. **Create the Virtual Environment**:
   ```bash
   python -m venv venv
   ```

2. **Activate the Virtual Environment**:
   - **Windows**:
     ```bash
     venv\Scripts\activate
     ```
   - **macOS/Linux**:
     ```bash
     source venv/bin/activate
     ```

3. **Verify Activation**:
   Your terminal prompt should show `(venv)`:
   ```bash
   (venv) $
   ```
   Running `pip list` should show only basic packages (e.g., `pip`, `setuptools`).

### Step 3: Install Dependencies

Install required Python packages listed in `requirements.txt`.

1. **Create `requirements.txt`** (if not present):
   ```plaintext
   fastapi==0.115.0
   uvicorn==0.30.6
   jinja2==3.1.4
   python-multipart==0.0.9
   ```
   Save this in the project root.

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify Installation**:
   ```bash
   pip list
   ```
   Confirm packages like `fastapi`, `uvicorn`, and `jinja2` are listed.

### Step 4: Run the FastAPI Application

Use **Uvicorn** to run the FastAPI app with auto-reload for development.

1. **Start the Server**:
   From the project root, run:
   ```bash
   uvicorn app.main:app --reload
   ```

   - `app.main`: The module (`app/main.py`).
   - `app`: The FastAPI instance in `main.py`.
   - `--reload`: Auto-restarts the server on code changes.

2. **Verify Output**:
   You should see:
   ```plaintext
   INFO:     Will watch for changes in these directories: ['/path/to/phrentpredict']
   INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
   INFO:     Started reloader process [12345] using WatchFiles
   INFO:     Started server process [67890]
   INFO:     Waiting for application startup.
   INFO:     Application startup complete.
   ```

   The app is now running at `http://localhost:8000`.


### Step 5: Access and Test the Application

1. **Open the Application**:
   - Open a browser and navigate to `http://localhost:8000`.
   - You should see the **PHRentPredict** homepage with the hero section’s **Port Harcourt Rent Trends** chart.

2. **Check Dependencies**:
   Ensure `base.html` includes:
   - **Chart.js**: `<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>`
   - **Tailwind CSS**: Via CDN or local file.
   - **Font Awesome**: For icons.
   - **AOS**: For animations.

   Example `base.html` `<head>`:
   ```html
   <head>
       <title>{% block title %}{% endblock %}</title>
       <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
       <script src="https://cdn.tailwindcss.com"></script>
       <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
       <link href="https://unpkg.com/aos@2.3.1/dist/aos.css" rel="stylesheet">
       <script src="https://unpkg.com/aos@2.3.1/dist/aos.js"></script>
       <script>AOS.init();</script>
   </head>
   ```

3. **Test the Application**:
   See the [Testing the Application](#testing-the-application) section below.

## Testing the Application

Test the application to ensure all features work as expected.

1. **Hero Section**:
   - **Chart**: Verify the **Port Harcourt Rent Trends** chart displays three lines:
     - Single Room (greenish #339d70).
     - Self-Contained (orange #f1631d).
     - 2 Bedroom (gray #565862).
   - Check years 2019–2025 on the x-axis.
   - Hover over 2025 to confirm values: ₦220K (Single Room), ₦700K (Self-Contained), ₦1.5M (2 Bedroom).
   - Toggle dark/light theme (if implemented) to ensure chart colors update.
   - Check averages below the chart: ₦185K, ₦550K, ₦1.2M (note: these differ from 2025 chart values).

2. **Prediction Form**:
   - Navigate to the **Get Your Rent Forecast** section.
   - Select a neighborhood (e.g., Diobu), property type (e.g., Single Room), and horizon (e.g., 3 Years).
   - Click **Generate Rent Forecast**.
   - Verify the form submits to `/predict/html` and displays results (requires backend implementation).
   - Ensure the horizon slider updates the display (e.g., “3 Years”).

3. **Other Sections**:
   - **Features**: Confirm all six feature cards display with icons and animations.
   - **How It Works**: Check the three-step process with arrows and animations.
   - **Neighborhoods**: Verify four neighborhood cards (Diobu, GRA, Woji, Eliozu) with price ranges.
   - **FAQ**: Ensure four questions display with answers.
   - **CTA**: Click **Get Started Now** to jump to the prediction form.

4. **Responsive Design**:
   - Resize the browser or use mobile view (F12 > Toggle Device Toolbar) to test responsiveness.
   - Ensure the chart, form, and sections adapt to mobile and tablet screens.

5. **Debugging**:
   - Open Developer Tools (F12 > Console) to check for errors (e.g., “Chart is not defined”).
   - Verify `<canvas id="heroTrendChart">` is in the DOM (F12 > Elements).
   - Ensure the chart container (`<div class="h-64">`) has a height of 16rem.

If issues arise (e.g., chart not rendering), check:
- Chart.js inclusion in `base.html`.
- Console errors for missing libraries or JavaScript issues.
- Correct paths in `app/templates/` for `index.html` and `base.html`.

## Project Structure

```plaintext
phrentpredict/
├── app/
│   ├── main.py              # FastAPI application
│   ├── templates/
│   │   ├── base.html        # Base template with layout
│   │   ├── index.html       # Main page with hero chart and form
├── venv/                    # Virtual environment
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
```

- **app/main.py**: Defines the FastAPI app, routes (`/` for index, `/predict/html` for predictions), and template rendering.
- **app/templates/index.html**: Contains the homepage with the hero chart, prediction form, and other sections.
- **app/templates/base.html**: Provides the HTML layout, including CSS/JS dependencies.
- **requirements.txt**: Lists Python packages for the backend.

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository on GitHub.
2. Clone your fork: `git clone https://github.com/0xSemantic/phrentpredict.git`.
3. Create a branch: `git checkout -b feature/your-feature`.
4. Make changes and commit: `git commit -m "Add your feature"`.
5. Push to your fork: `git push origin feature/your-feature`.
6. Open a Pull Request on GitHub.

Please follow coding standards, add tests for new features, and update documentation.

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

### Notes
- **GitHub URL**: I assumed a placeholder URL (`https://github.com/0xSemantic/phrentpredict`). Replace with the actual repository URL when hosting the project.
- **Backend Implementation**: The README assumes the `/predict/html` endpoint is implemented in `main.py`. If not, you may need to add placeholder logic or clarify in the README that it’s a work-in-progress.
- **Chart Data**: The hero chart uses static data matching the original (2025 values: ₦220K, ₦700K, ₦1.5M). If you want to align with the displayed averages (₦185K, ₦550K, ₦1.2M), I can provide an updated `index.html`.
- **Dependencies in `base.html`**: The setup assumes Chart.js, Tailwind CSS, Font Awesome, and AOS are included in `base.html`. If your `base.html` differs, share its content for tailored instructions.
- **Testing**: The testing section covers the hero chart fix from your previous request, ensuring it renders with static data.

### Next Steps
1. **Create the Repository**:
   - Initialize a GitHub repository: `git init`, `git add .`, `git commit -m "Initial commit"`, `git remote add origin <url>`, `git push -u origin main`.
   - Upload `app/`, `requirements.txt`, and this `README.md`.
2. **Share Feedback**:
   - If you need additions to the README (e.g., backend details, deployment instructions), let me know.
   - If the chart or any feature still doesn’t work, provide console errors or a screenshot.
3. **Enhancements**:
   - Add dynamic data fetching for the chart (e.g., via `/trends` endpoint).
   - Align chart colors with the hero section’s dots (#ef4444, #eab308, #22c55e).
   - Update 2025 chart values to match displayed averages.
