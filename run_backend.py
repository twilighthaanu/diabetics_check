import os
import sys
import logging
from src.api.app import app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('backend.log')
    ]
)
logger = logging.getLogger(__name__)

def main():
    try:
        # Create necessary directories
        upload_dir = os.path.join('data', 'uploads')
        static_dir = 'static'
        
        for directory in [upload_dir, static_dir]:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Ensured directory exists: {os.path.abspath(directory)}")
        
        # Start the Flask app
        host = '0.0.0.0'  # Listen on all network interfaces
        port = 5000
        
        logger.info(f"Starting Flask server on http://{host}:{port}")
        logger.info("Available routes:")
        for rule in app.url_map.iter_rules():
            logger.info(f"  {rule.endpoint}: {rule.rule} ({','.join(rule.methods)})")
            
        app.run(host=host, port=port, debug=True, use_reloader=False)
        
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
