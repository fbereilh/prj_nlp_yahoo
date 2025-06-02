from fasthtml.common import *
from model import load_model, predict_text, MODEL_CONFIG
from database import add_prediction, get_recent_predictions, delete_prediction, Prediction
from search import initialize_search, search_predictions
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastHTML app with Pico CSS and Chart.js
app, rt = fast_app(
    pico=True,  # Enable Pico CSS
    hdrs=(
        # Add Chart.js for visualization
        Script(src="https://cdn.jsdelivr.net/npm/chart.js"),
        # Add direct style for margin
        Style("body { margin-top: 3rem !important; }"),
        # Link to external CSS file
        Link(rel="stylesheet", href="/static/styles.css", type="text/css")
    )
)

# Global flag to track model status
model_loaded = False

def render_prediction(pred: Prediction):
    """Helper to render a prediction as an Article with improved layout"""
    import json
    
    probabilities = json.loads(pred.probabilities)
    chart_id = f"chart-{pred.id}"
    pred_id = f"prediction-{pred.id}"
    
    # Format the timestamp to show only up to seconds
    timestamp = datetime.strptime(pred.created_at, "%Y-%m-%d %H:%M:%S.%f")
    formatted_time = timestamp.strftime("%Y-%m-%d %H:%M:%S")
    
    chart_js = f"""
    function initChart_{pred.id}() {{
        // Always clean up existing chart first
        let existingChart = Chart.getChart('{chart_id}');
        if (existingChart) {{
            existingChart.destroy();
        }}
        
        const ctx = document.getElementById('{chart_id}');
        if (!ctx) return;  // Exit if canvas not found
        
        // Set fixed dimensions for the canvas
        ctx.style.height = '300px';
        ctx.style.width = '100%';
        
        new Chart(ctx, {{
            type: 'bar',
            data: {{
                labels: {list(probabilities.keys())},
                datasets: [{{
                    label: 'Probability',
                    data: {list(probabilities.values())},
                    backgroundColor: 'rgba(54, 162, 235, 0.5)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 1
                }}]
            }},
            options: {{
                indexAxis: 'y',
                maintainAspectRatio: false,
                responsive: true,
                plugins: {{
                    legend: {{ display: false }},
                }},
                scales: {{
                    x: {{
                        beginAtZero: true,
                        max: 1,
                        title: {{
                            display: true,
                            text: 'Probability'
                        }}
                    }},
                    y: {{
                        ticks: {{
                            autoSkip: false,  // Prevent skipping labels
                            padding: 10,      // Add some padding
                            font: {{
                                size: 11     // Slightly smaller font for better fit
                            }}
                        }}
                    }}
                }},
                animation: {{
                    duration: 0  // Disable animations for smoother updates
                }}
            }}
        }});
    }}

    // Clean up function for this specific chart
    function cleanupChart_{pred.id}() {{
        let chart = Chart.getChart('{chart_id}');
        if (chart) {{
            chart.destroy();
        }}
    }}

    // Initialize on DOM content loaded and after HTMX swaps
    document.addEventListener('DOMContentLoaded', initChart_{pred.id});
    document.addEventListener('htmx:afterSettle', function(evt) {{
        if (document.getElementById('{chart_id}')) {{
            initChart_{pred.id}();
        }}
    }});

    // Handle HTMX events
    document.addEventListener('htmx:beforeSwap', function(evt) {{
        if (evt.detail.target && evt.detail.target.contains(document.getElementById('{chart_id}'))) {{
            cleanupChart_{pred.id}();
        }}
    }});
    """
    
    return Article(
        Button(
            "×",
            cls="delete-btn",
            hx_delete=f"/predict/{pred.id}",
            hx_target=f"#prediction-{pred.id}",
            hx_swap="outerHTML"
        ),
        Div(  # Left side
            Div(  # Top part - Question
                P(pred.text, cls="prediction-text"),
                cls="prediction-question"
            ),
            Div(  # Bottom part - Prediction data
                P(Strong("Category: "), pred.category, cls="prediction-meta"),
                P(Strong("Confidence: "), f"{pred.confidence:.1%}", cls="prediction-meta"),
                P(Strong("Predicted at: "), formatted_time, cls="prediction-meta"),
                cls="prediction-data"
            ),
            cls="prediction-left-side"
        ),
        Div(  # Right side - Chart
            Canvas(id=chart_id),
            cls="prediction-right-side"
        ),
        Script(chart_js),
        cls="prediction-item",
        id=pred_id
    )

def render_recent_predictions():
    """Helper to render the recent predictions section"""
    recent = get_recent_predictions(5)  # Already returns in reverse chronological order
    return Div(
        H2("Recent Predictions"),
        Div(
            *[render_prediction(pred) for pred in recent],
            id="predictions-list",
            hx_swap_oob="true"  # Ensure proper swap behavior
        ),
        cls="predictions-list"
    )

def render_search_results(results):
    """Helper to render search results"""
    if not results:
        return Div(
            P("No similar predictions found.", cls="info-message"),
            id="search-results"
        )
    
    return Div(
        H2("Search Results"),
        Div(
            *[
                Div(
                    render_prediction(pred),
                    P(f"Similarity Score: {sim * 100:.1f}%", cls="similarity-score"),
                    cls="search-result-item"
                ) 
                for pred, sim in results
            ],
            id="search-results"
        ),
        cls="search-results"
    )

@rt
def index():
    """Render the main page with the prediction form and recent predictions"""
    github_svg = """
        <svg viewBox="0 0 24 24" aria-hidden="true" fill="currentColor">
            <path fill-rule="evenodd" clip-rule="evenodd" d="M12 2C6.477 2 2 6.463 2 11.97c0 4.404 2.865 8.14 6.839 9.458.5.092.682-.216.682-.48 0-.236-.008-.864-.013-1.695-2.782.602-3.369-1.337-3.369-1.337-.454-1.151-1.11-1.458-1.11-1.458-.908-.618.069-.606.069-.606 1.003.07 1.531 1.027 1.531 1.027.892 1.524 2.341 1.084 2.91.828.092-.643.35-1.083.636-1.332-2.22-.251-4.555-1.107-4.555-4.927 0-1.088.39-1.979 1.029-2.675-.103-.252-.446-1.266.098-2.638 0 0 .84-.268 2.75 1.022A9.607 9.607 0 0112 6.82c.85.004 1.705.114 2.504.336 1.909-1.29 2.747-1.022 2.747-1.022.546 1.372.202 2.386.1 2.638.64.696 1.028 1.587 1.028 2.675 0 3.83-2.339 4.673-4.566 4.92.359.307.678.915.678 1.846 0 1.332-.012 2.407-.012 2.734 0 .267.18.577.688.48 3.97-1.32 6.833-5.054 6.833-9.458C22 6.463 17.522 2 12 2z"></path>
        </svg>
    """
    
    return Titled(
        "Yahoo Text Classifier",  # This sets the page title
        Div(  # Container wrapper
            Main(  # Use Main instead of Div for better semantics
                Div(  # Header container
                    A(
                        NotStr(github_svg),
                        href="https://github.com/fbereilh/prj_nlp_yahoo",
                        target="_blank",
                        cls="github-link",
                        title="View on GitHub"
                    ),
                    cls="header-container"
                ),
                P("Enter your text below to classify it into one of the Yahoo categories."),
                # Add search form at the top
                Form(
                    Div(
                        Label("Search predictions:"),
                        Input(
                            id="search-input",
                            name="query",
                            placeholder="Search similar questions...",
                            hx_post="/search",
                            hx_trigger="keyup changed delay:500ms",
                            hx_target="#predictions-list",
                            hx_swap="innerHTML"
                        ),
                        cls="input-group"
                    ),
                    cls="search-form"
                ),
                Form(
                    Div(
                        Label("Text to classify:"),
                        Textarea(
                            id="text-input",
                            name="text",
                            placeholder="Enter text to classify...",
                            required=True
                        ),
                        cls="input-group"
                    ),
                    Button("Classify", type="submit"),
                    id="predict-form",
                    hx_post="/predict",
                    hx_target="#predictions-list",
                    hx_swap="afterbegin"
                ),
                # Recent predictions section
                Div(
                    H2("Predictions"),
                    Div(
                        *[render_prediction(pred) for pred in get_recent_predictions(5)],
                        id="predictions-list",
                        hx_swap_oob="true"  # Ensure proper swap behavior
                    ),
                    cls="predictions-list"
                )
            ),
            cls="container"
        )
    )

@rt("/predict")
def post(text: str):
    """Handle prediction and return formatted results"""
    global model_loaded
    
    if not model_loaded:
        logger.warning("Prediction attempted but model not loaded")
        return Article(
            P("Error: Model not loaded. Please try again in a few moments.", 
              role="alert",
              cls="error-message")
        )
    
    if not text:
        logger.warning("Empty text submitted for prediction")
        return Article(
            P("Error: No text provided", 
              role="alert",
              cls="error-message")
        )
    
    logger.info(f"Making prediction for text: {text[:50]}...")
    result = predict_text(text)
    
    if "error" in result:
        logger.error(f"Prediction error: {result['error']}")
        return Article(
            P(f"Error: {result['error']}", 
              role="alert",
              cls="error-message")
        )

    logger.info(f"Prediction successful: {result['prediction']}")
    # Save prediction to database
    pred = add_prediction(text, result)
    
    # Return:
    # 1. Clear the textarea (out-of-band swap)
    # 2. Just return the new prediction at the top
    return (
        Textarea(
            id="text-input",
            name="text",
            rows=5,
            placeholder="Enter text to classify...",
            required=True,
            hx_swap_oob="true"
        ),
        render_prediction(pred)  # Only return the new prediction
    )

@rt("/predict/{id}")
def delete(id: str):
    """Delete a prediction"""
    try:
        success = delete_prediction(id)
        if not success:
            return Article(
                P("Error: Prediction not found", role="alert", cls="error-message")
            )
        
        # Return empty string since we're using outerHTML swap
        return ""
        
    except Exception as e:
        logger.error(f"Error deleting prediction {id}: {str(e)}")
        return Article(
            P(f"Error deleting prediction: {str(e)}", role="alert", cls="error-message")
        )

@rt("/search")
def post(query: str):
    """Handle search and return formatted results"""
    if not query:
        # If no query, show all predictions
        return Div(
            *[render_prediction(pred) for pred in get_recent_predictions(5)],
            id="predictions-list"
        )
    
    logger.info(f"Searching for: {query[:50]}...")
    results = search_predictions(query)
    
    # Return the predictions list with search results
    return Div(
        *[
            Div(
                P(f"Similarity: {sim:.1%}", cls="similarity-score"),
                render_prediction(pred),
                cls="prediction-with-score"
            ) 
            for pred, sim in results
        ],
        id="predictions-list"
    )

@rt("/static/styles.css")
def get():
    """Serve the CSS file"""
    return Response(
        content=open("static/styles.css").read(),
        media_type="text/css"
    )

@rt("/health")
def get():
    """Health check endpoint"""
    if not model_loaded:
        return Response(
            content={"status": "error", "message": "Model not loaded"},
            status_code=503
        )
    return {"status": "healthy", "message": "Service is running"}

# Startup event to load model
@app.on_event("startup")
async def startup_event():
    """Initialize the model and search index on startup"""
    global model_loaded
    try:
        logger.info("Starting application initialization...")
        
        # Load model with optimized settings first
        logger.info("Starting model initialization...")
        import torch
        if torch.cuda.is_available():
            # Use GPU if available
            device = "cuda"
            logger.info("CUDA is available, using GPU")
            torch.cuda.empty_cache()
        else:
            # Use CPU with optimized settings
            device = "cpu"
            logger.info("CUDA not available, using CPU")
            import os
            n_threads = min(os.cpu_count() or 1, 4)  # Use at most 4 threads
            torch.set_num_threads(n_threads)
            logger.info(f"Set CPU threads to {n_threads}")
        
        if not load_model():
            raise RuntimeError("Failed to load model")
            
        model_loaded = True
        logger.info("Model initialization completed successfully")
        
        # Now initialize search after model is loaded
        logger.info("Initializing search functionality...")
        try:
            if not initialize_search():
                logger.warning("Search initialization returned False - this might mean no predictions in database")
            logger.info("Search initialization completed")
        except Exception as e:
            logger.error(f"Search initialization failed: {str(e)}")
            # Don't fail startup if search fails - it will reinitialize when data is available
            
        logger.info("Application startup completed successfully")
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}", exc_info=True)
        model_loaded = False
        raise  # Re-raise the exception to ensure the app doesn't start with failed initialization

if __name__ == "__main__":
    serve() 