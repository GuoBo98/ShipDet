from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import LoadAnnotations

@PIPELINES.register_module()
class LoadAnnotationsOBB(LoadAnnotations):
    def __init__(self,
                 with_obb=False,
                #  with_annotation_type=False,
                #  with_rpnsemi_type=False,
                #  with_rbbox_reg_type=False,
                 **kwargs):
        super(LoadAnnotationsOBB,self).__init__(**kwargs)
        self.with_obb = with_obb
        self.with_annotation_type = with_annotation_type
        self.with_rpnsemi_type = with_rpnsemi_type
        self.with_rbbox_reg_type = with_rbbox_reg_type

    def _load_obb(self, results):
        results['gt_obbs'] = results['ann_info']['obbs']
        results['obb_fields'].append('gt_obbs')

        return results
    def __call__(self, results):
        results = super().__call__(results)
        if self.with_obb:
            results = self._load_obb(results)
        # if self.with_annotation_type:
        #     results = self._load_annotation_type(results)
        # if self.with_rpnsemi_type:
        #     results = self._load_rpnsemi_type(results)
        # if self.with_rbbox_reg_type:
        #     results = self._load_rbbox_reg_type(results)
        return results
    def __repr__(self):
        """str: Return a string that describes the module."""
        indent_str = '    '
        repr_str = self.__class__.__name__ + '(\n'
        repr_str += f'{indent_str}with_obb={self.with_obb})'
        # repr_str += f'{indent_str}with_annotation_type={self.with_annotation_type})'
        # repr_str += f'{indent_str}with_rpnsemi_type={self.with_rpnsemi_type})'
        # repr_str += f'{indent_str}with_rbbox_reg_type={self.with_rbbox_reg_type})'
        return repr_str