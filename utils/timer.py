
import time
import logging

class Timer:
    """Timer class for recording the time taken by each step"""
    
    def __init__(self):
        self.start_time = None
        self.step_times = {}
        self.current_step = None
        self.step_start_times = {}
    
    def start(self, step_name):
        """Start timing a new step"""
        if self.current_step:
            self.end()  # End the current step if one is running
            
        self.current_step = step_name
        self.start_time = time.time()
        self.step_start_times[step_name] = self.start_time
        logging.info(f"Step started: {step_name}")
    
    def end(self):
        """End the current step timing"""
        if self.current_step and self.start_time:
            elapsed = time.time() - self.start_time
            self.step_times[self.current_step] = elapsed
            logging.info(f"Step completed: {self.current_step} | Time taken: {elapsed:.2f} seconds")
            self.current_step = None
            self.start_time = None
    
    def get_step_time(self, step_name):
        """Get the time taken for a specific step"""
        return self.step_times.get(step_name, 0.0)
    
    def total_time(self):
        """Calculate total time taken"""
        return sum(self.step_times.values())
    
    def log_summary(self):
        """Log a summary of all step timings"""
        logging.info("\n===== Step Timing Summary =====")
        for step, elapsed in self.step_times.items():
            logging.info(f"{step}: {elapsed:.2f} seconds")
        logging.info(f"Total time: {self.total_time():.2f} seconds")
        logging.info("================================\n")
        
        # Return summary string, useful for saving to a file
        summary_lines = ["===== Step Timing Summary ====="]
        for step, elapsed in self.step_times.items():
            summary_lines.append(f"{step}: {elapsed:.2f} seconds")
        summary_lines.append(f"Total time: {self.total_time():.2f} seconds")
        summary_lines.append("================================\n")
        return "\n".join(summary_lines)
