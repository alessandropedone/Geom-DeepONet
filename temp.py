from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
import time

with Progress(
    TextColumn("[bold blue]{task.description}"),
    BarColumn(bar_width=None, complete_style="green", finished_style="bright_green", pulse_style="yellow"),
    "[progress.percentage]{task.percentage:>3.0f}%",
    TimeElapsedColumn(),
    TimeRemainingColumn()
) as progress:
    task = progress.add_task("Downloading...", total=100)
    
    while not progress.finished:
        progress.update(task, advance=1)
        time.sleep(0.05)
