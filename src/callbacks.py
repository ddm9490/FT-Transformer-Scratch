# callbacks.py
import lightning as L
from rich.console import Console
from rich.table import Table
from rich import box

class RichHyperparametersSummary(L.Callback):
    def __init__(self, title: str = "Hyperparameter Summary"):
        super().__init__()
        self.title = title

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        # Multi-GPU(DDP) 환경에서 0번 프로세스만 출력
        if not trainer.is_global_zero:
            return

        console = Console()
        table = Table(
            title=f"\n[bold cyan]=== {self.title} ===[/bold cyan]",
            box=box.HEAVY_HEAD,
            header_style="bold magenta"
        )
        table.add_column("Category", style="cyan", no_wrap=True)
        table.add_column("Parameter", style="white")
        table.add_column("Value", style="green")

        # 1. Model의 hparams 수집
        if hasattr(pl_module, "hparams"):
            for k, v in dict(pl_module.hparams).items():
                if k != "_instantiator":
                    table.add_row("Model", str(k), str(v))

        # 2. DataModule의 hparams 수집
        if trainer.datamodule and hasattr(trainer.datamodule, "hparams"):
            for k, v in dict(trainer.datamodule.hparams).items():
                if k != "_instantiator":
                    table.add_row("Data", str(k), str(v))

        console.print(table)

