from cont_chir_extrap import *

mu = 2.0
b = bag_fits(bag_ensembles, obj='bag')
laxis = {1:1, 2:1, 3:1, 4:0, 5:0}
record_bag = {}

for op_idx, op in enumerate(b.operators):
    title = r'$B_'+str(op_idx+1)+r' = B_'+str(op_idx+1)+'^{phys}'
    title += r' + \alpha\,a^2 + \beta\,\left(m_\pi^2-m_\pi^{2,phys}\right)$'
    filename = f'/Users/rajnandinimukherjee/Desktop/draft_plots/linear/bag_fits_B{op_idx+1}_{int(mu*10)}.pdf'
    record_bag[f'B{op_idx+1}'] = b.fit_operator(mu, op, rotate=NPR_to_SUSY, obj='bag',
                                                title=title, chiral_extrap=True, rescale=True,
                                                plot=True, open=True, figsize=(10,3),
                                                label='', legend_axis=laxis[op_idx+1],
                                                filename=filename)


r = bag_fits(bag_ensembles, obj='ratio')
record_ratio = {}
for op_idx, op in enumerate(r.operators):
    title = r'$R_'+str(op_idx+2)+r' = R_'+str(op_idx+2)+'^{phys}'
    title += r' + \alpha\,a^2 + \beta\,\left(m_\pi^2-m_\pi^{2,phys}\right)$'
    filename = f'/Users/rajnandinimukherjee/Desktop/draft_plots/linear/ratio_fits_R{op_idx+2}_{int(mu*10)}.pdf'
    record_ratio[f'R{op_idx+2}'] = r.fit_operator(mu, op, rotate=NPR_to_SUSY,
                                                  obj='ratio', title=title,
                                                  chiral_extrap=True, plot=True,
                                                  open=True, figsize=(10,3), label='',
                                                  legend_axis=0, filename=filename)


def results_tables(record, mu, **kwargs):
    return 0
