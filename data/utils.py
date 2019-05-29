import json
import os
from os.path import join, basename, dirname, isfile, exists

class Field:
    def __init__(self):
        self.field_name = ''
        self.feature_num = 0
        self.bound_field_name = ''
        self.bound_field_id = -1
        self.inv_bound_field_id = -1


class IdMap:
    def __init__(self):
        self.field_num = 0
        self.fields = []

    def load(self, fin):
        self.field_num, dump = [int(val) for val in fin.readline().strip().split('\t')]
        field_name_id_dict = {}
        for i in xrange(self.field_num):
            tokens = fin.readline().strip().split('\t')
            field = Field()
            field.field_name = tokens[0]
            field.feature_num = int(tokens[1])
            if len(tokens) > 2:
                field.bound_field_name = tokens[2]
            self.fields.append(field)
            field_name_id_dict[field.field_name] = i

            for j in xrange(field.feature_num):
                fin.readline()

        for i in xrange(self.field_num):
            if self.fields[i].bound_field_name != '':
                self.fields[i].bound_field_id = field_name_id_dict[self.fields[i].bound_field_name]
                self.fields[field_name_id_dict[self.fields[i].bound_field_name]].inv_bounf_field_id = i


class ListIdMap:
    def __init__(self):
        self.item_feature_id_map = IdMap()
        self.user_feature_id_map = IdMap()
        self.recent_item_feature_id_map = IdMap()
        self.item_label_id_map = IdMap()
        self.list_label_id_map = IdMap()

    def load_feature_id_map(self, infile):
        with open(infile, 'r') as fin:
            self.item_feature_id_map.load(fin)
            self.user_feature_id_map.load(fin)
            self.recent_item_feature_id_map.load(fin)

    def load_label_id_map(self, infile):
        with open(infile, 'r') as fin:
            self.item_label_id_map.load(fin)
            self.list_label_id_map.load(fin)

    def get_all_field_num(self):
        return self.item_feature_id_map.field_num + \
                self.user_feature_id_map.field_num + \
                self.recent_item_feature_id_map.field_num + \
                self.item_label_id_map.field_num + \
                self.list_label_id_map.field_num

    def get_all_fields(self):
        return self.item_feature_id_map.fields + \
                self.user_feature_id_map.fields + \
                self.recent_item_feature_id_map.fields + \
                self.item_label_id_map.fields + \
                self.list_label_id_map.fields
