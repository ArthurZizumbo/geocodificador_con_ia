# scripts/evaluate_model.py
import polars as pl
import requests
import typer
from pathlib import Path
from haversine import haversine, Unit
import numpy as np

app = typer.Typer()
API_URL = "http://localhost:8000/geocodificar"

@app.command()
def evaluate(eval_path: Path = typer.Argument(..., help="Ruta al CSV con los datos de evaluación.")):
    """Evalúa el rendimiento del modelo de geocodificación con un reporte detallado."""
    if not eval_path.exists():
        typer.secho(f"❌ Error: No se encontró el archivo de evaluación en {eval_path}", fg=typer.colors.RED); raise typer.Exit(1)
            
    df_eval = pl.read_csv(eval_path)
    
    # Crear el mapeo de entidades usando polars
    entity_mapping = {15: 'MEX', 9: 'CMX', 31: 'YUC', 7: 'CHS'}
    df_eval = df_eval.with_columns(
        (pl.col('entidad').replace(entity_mapping) + '-' + pl.col('id_domicilio').cast(pl.Utf8))
        .alias('id_global_domicilio_real')
    )

    distances = []
    reciprocal_ranks = []

    typer.echo(f"\n▶️  Iniciando evaluación detallada de {df_eval.height} casos de prueba...\n")
    
    for row in df_eval.iter_rows(named=True):
        typer.echo(f"--- Caso de Prueba: '{row['query']}' ---")
        expected_id = row['id_global_domicilio_real']
        typer.echo(f"  - Esperado: {expected_id}")

        payload = {"query": row['query'], "top_k": 5}
        try:
            response = requests.post(API_URL, json=payload)
            response.raise_for_status()
            results = response.json()["results"]
        except requests.exceptions.RequestException as e:
            typer.secho(f"  - ❌ Error en API: {e}", fg=typer.colors.RED); continue

        if not results:
            typer.secho("  - ❌ Fallo: La API no devolvió resultados.", fg=typer.colors.RED)
            reciprocal_ranks.append(0)
            continue

        top_ids = [res["id_global_domicilio"] for res in results]
        top_result_id = top_ids[0]
        typer.echo(f"  - Obtenido:  {top_result_id} (Score: {results[0]['score']:.4f})")
        
        # --- Lógica de Reporte Detallado ---
        if top_result_id == expected_id:
            typer.secho("  - ✅ Acierto en @1", fg=typer.colors.GREEN)
        else:
            typer.secho("  - ❌ Fallo en @1", fg=typer.colors.RED)
            if expected_id in top_ids:
                rank = top_ids.index(expected_id) + 1
                typer.echo(f"    (La respuesta correcta estaba en la posición #{rank})")
            else:
                typer.echo("    (La respuesta correcta no apareció en el top 5)")
        
        # --- Cálculo de métricas (sin cambios) ---
        try:
            rank = top_ids.index(expected_id) + 1
            reciprocal_ranks.append(1 / rank)
        except ValueError:
            reciprocal_ranks.append(0)

        if top_result_id == expected_id:
            pred_coords = (results[0].get('latitud'), results[0].get('longitud'))
            real_coords = (row['lat_real'], row['lon_real'])
            if pred_coords[0] is not None and pred_coords[1] is not None:
                dist = haversine(pred_coords, real_coords, unit=Unit.METERS)
                distances.append(dist)
                typer.echo(f"  - Distancia: {dist:.2f} metros")
        
        typer.echo("-" * 40)


    # Reporte final (sin cambios)
    accuracy_at_1 = sum(1 for r in reciprocal_ranks if r == 1) / len(reciprocal_ranks)
    mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0
    avg_distance_error = np.mean(distances) if distances else float('inf')

    typer.echo("\n--- 📈 Reporte de Evaluación Final ---")
    typer.secho(f"Accuracy@1: {accuracy_at_1:.2%}", fg=typer.colors.GREEN)
    typer.secho(f"Mean Reciprocal Rank (MRR): {mrr:.4f}", fg=typer.colors.GREEN)
    typer.secho(f"Error Promedio de Distancia (en aciertos @1): {avg_distance_error:.2f} metros", fg=typer.colors.CYAN)
    typer.echo("------------------------------------")

if __name__ == "__main__":
    app()