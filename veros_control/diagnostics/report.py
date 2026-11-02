"""Annual maps, multiyear series and training loss from paired monthly archives."""
import argparse
from pathlib import Path
from types import SimpleNamespace
from veros_control.diagnostics.annual import run as compute
from veros_control.diagnostics.plots import run as plot
from veros_control.diagnostics.series import common_support, long_series


def run(args):
    from pypdf import PdfWriter
    root=Path(args.run_dir);assets=Path(args.glorys_dir)
    years=sorted(set(args.years))
    support=root/'common_support.npz'
    common_support(root,assets,years,support)
    products=[];pdfs=[]
    for year in years:
        output=root/f'evaluation_{year}';output.mkdir(exist_ok=True)
        product=output/'annual.nc';products.append(product)
        compute(SimpleNamespace(run_dir=str(root),glorys_dir=str(assets),year=year,
            model_thermodynamics=args.model_thermodynamics,output=str(product),common_mask_file=str(support)))
        plot(SimpleNamespace(input=str(product),output_dir=str(output)))
        writer=PdfWriter()
        for name in ('time_series','sst','sss','ssh','mld_winter','mld_winter_no_nn',
                     'mld_summer','mld_summer_no_nn','zonal_temp','zonal_salt','overturning'):
            writer.append(str(output/f'{name}.pdf'))
        writer.write(str(output/'evaluation.pdf'));writer.close()
        pdfs.append(output/'evaluation.pdf')
    series=root/'timeseries';series.mkdir(exist_ok=True)
    long_series(products,series)
    pdfs.insert(0,series/'time_series_early_years.pdf')
    if (root/'checkpoint.pkl').exists():
        from veros_control.diagnostics.rollout_loss import compute as loss
        loss(root,assets,years);pdfs.append(root/'rollout_loss.pdf')
    import xarray as xr
    with xr.open_dataset(root/'trained.averages.nc',decode_times=False) as ds:
        has_corrections='dtemp_nn' in ds
    if has_corrections:
        from veros_control.diagnostics.corrections import plot_corrections
        plot_corrections(root);pdfs.append(root/'corrections.pdf')
    if (root/'baseline_stability.csv').exists():
        from veros_control.diagnostics.stability import plot_stability
        plot_stability(root);pdfs.append(root/'stability.pdf')
    writer=PdfWriter()
    for pdf in pdfs:writer.append(str(pdf))
    writer.write(str(root/'train_diags.pdf'));writer.close()
    return root/'train_diags.pdf'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',required=True)
    parser.add_argument('--glorys-dir',required=True,help='Already remapped monthly GLORYS files')
    parser.add_argument('--years',required=True,type=int,nargs='+')
    parser.add_argument('--model-thermodynamics',choices=('ct-sa','pt-sp'),default='ct-sa')
    print(run(parser.parse_args()))


if __name__=='__main__':main()
