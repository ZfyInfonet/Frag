import networkx as nx
import numpy as np


class SFC:
    def __init__(self, sfc_id, vnf_list, virtual_vnf_list, vnf_link_list, lifetime):
        self.sfc_id = sfc_id
        self.vnf_list = vnf_list
        self.vnf_link_list = vnf_link_list
        self.virtual_vnf_list = virtual_vnf_list
        self.is_mapped = False
        self.lifetime = lifetime
        self.arrive_time = None

    def updateReq(self, t):
        pass


class VNF:
    def __init__(self, sfc_id, vnf_id, is_entrance, is_exit, trace_cpu = None,
                 trace_mem = None, node_id = None, delay_proc = 0):

        self.sfc_id = sfc_id
        self.vnf_id = vnf_id
        self.is_entrance = is_entrance
        self.is_exit = is_exit

        self.trace_cpu = trace_cpu
        self.trace_mem = trace_mem
        self.node_id = node_id
        self.delay_proc = delay_proc
        self.is_migrating = False
        self.upstream_link = None
        self.downstream_link = []
        self.links = []

        if trace_cpu is not None:
            self.req_cpu = trace_cpu[0]
        if trace_mem is not None:
            self.req_mem = trace_mem[0]


class VNFLink:
    def __init__(self, sfc_id, source_vnf_id, destination_vnf_id, vnfs, delay_limit, trace_bw = None):
        self.sfc_id = sfc_id
        self.ends = (source_vnf_id, destination_vnf_id)
        self.vnfs = vnfs
        self.edges = []
        self.trace_bw = trace_bw
        self.is_mapped = False
        self.delay_limit = delay_limit
        if trace_bw is not None:
            self.req_bw = trace_bw[0]
