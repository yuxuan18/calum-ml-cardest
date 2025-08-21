import copy
from dsb_utlities.NeuroCDF_helpers import *
import random

OPS = ['<', '<=', '>', '>=', '=']
IN_BUCKETS = 100

def get_table_info(file_path='./queries/column_min_max_vals_tpcds.csv'):
	lines = open(file_path, 'r').readlines()

	table_size_list = []
	table2id = {}
	col2id = {}
	global_col2id = {}
	col2minmax = {}
	join_col_list = []

	join_cols = ['date_dim.d_week_seq', 'web_returns.wr_order_number', 'catalog_sales.cs_sold_time_sk', 'catalog_sales.cs_bill_addr_sk', 'catalog_sales.cs_catalog_page_sk', 'web_sales.ws_ship_mode_sk', 'catalog_sales.cs_ship_date_sk', 'store_returns.sr_cdemo_sk', 'customer_address.ca_zip', 'date_dim.d_date', 'customer.c_customer_sk', 'web_sales.ws_web_site_sk', 'catalog_sales.cs_ship_addr_sk', 'store_sales.ss_cdemo_sk', 'store_returns.sr_ticket_number', 'web_sales.ws_sold_date_sk', 'catalog_sales.cs_promo_sk', 'customer.c_first_shipto_date_sk', 'web_returns.wr_reason_sk', 'catalog_returns.cr_call_center_sk', 'store_returns.sr_returned_date_sk', 'web_sales.ws_ship_hdemo_sk', 'store.s_zip', 'web_sales.ws_bill_addr_sk', 'catalog_sales.cs_bill_hdemo_sk', 'inventory.inv_quantity_on_hand', 'reason.r_reason_sk', 'web_sales.ws_web_page_sk', 'store_returns.sr_item_sk', 'customer_demographics.cd_marital_status', 'inventory.inv_warehouse_sk', 'web_site.web_site_sk', 'store_sales.ss_promo_sk', 'catalog_returns.cr_returning_customer_sk', 'store_sales.ss_store_sk', 'store_returns.sr_customer_sk', 'warehouse.w_warehouse_sk', 'customer.c_birth_country', 'call_center.cc_call_center_sk', 'household_demographics.hd_demo_sk', 'item.i_item_id', 'store_returns.sr_reason_sk', 'inventory.inv_item_sk', 'web_returns.wr_returning_cdemo_sk', 'catalog_sales.cs_ship_customer_sk', 'time_dim.t_time_sk', 'web_returns.wr_returned_date_sk', 'web_sales.ws_bill_customer_sk', 'web_sales.ws_promo_sk', 'catalog_sales.cs_bill_customer_sk', 'store_sales.ss_addr_sk', 'web_page.wp_web_page_sk', 'store_sales.ss_hdemo_sk', 'inventory.inv_date_sk', 'web_returns.wr_item_sk', 'ship_mode.sm_ship_mode_sk', 'catalog_sales.cs_order_number', 'web_returns.wr_returning_addr_sk', 'store.s_store_sk', 'income_band.ib_income_band_sk', 'item.i_item_sk', 'catalog_sales.cs_bill_cdemo_sk', 'web_returns.wr_refunded_addr_sk', 'web_sales.ws_order_number', 'customer.c_current_cdemo_sk', 'catalog_sales.cs_call_center_sk', 'catalog_sales.cs_warehouse_sk', 'store_sales.ss_customer_sk', 'catalog_returns.cr_returned_date_sk', 'date_dim.d_date_sk', 'customer.c_current_addr_sk', 'promotion.p_promo_sk', 'customer_demographics.cd_education_status', 'store_sales.ss_sold_date_sk', 'customer.c_first_sales_date_sk', 'web_sales.ws_sold_time_sk', 'store_sales.ss_sold_time_sk', 'customer.c_current_hdemo_sk', 'catalog_sales.cs_quantity', 'catalog_returns.cr_order_number', 'store_returns.sr_store_sk', 'customer_address.ca_country', 'web_sales.ws_ship_addr_sk', 'catalog_sales.cs_sold_date_sk', 'customer_address.ca_address_sk', 'web_sales.ws_item_sk', 'catalog_sales.cs_item_sk', 'customer_demographics.cd_demo_sk', 'store_sales.ss_ticket_number', 'catalog_page.cp_catalog_page_sk', 'web_sales.ws_ship_date_sk', 'web_sales.ws_warehouse_sk', 'catalog_returns.cr_item_sk', 'household_demographics.hd_income_band_sk', 'web_returns.wr_refunded_cdemo_sk', 'date_dim.d_month_seq', 'catalog_returns.cr_returning_addr_sk', 'catalog_sales.cs_ship_mode_sk', 'store_sales.ss_item_sk', 'item.i_manufact_id']
	col_count = 0
	global_count = 0

	for line in lines[1:]:
		parts = line.strip().split(',')

		col = parts[0]
		table = col.split('.')[0]
		min_v = int(float(parts[1]))
		max_v = int(float(parts[2]))
		card = int(float(parts[3]))

		col2minmax[col] = [min_v, max_v + 1]

		if table not in table2id:
			col_count = 0
			table2id[table] = len(table2id)
			table_size_list.append(card)

		col2id[col] = col_count
		global_col2id[col] = global_count
		if col in join_cols:
			join_col_list.append(col_count)

		col_count += 1
		global_count += 1

	return table2id, table_size_list, col2minmax, col2id, join_col_list, global_col2id

def normalize_val(val, min_v, max_v):
	return float(val - min_v) / (max_v - min_v)

def read_query_file(table2id, col2minmax, col2id, global_col2id, num_q=30000, test_size=0, file_path='./queries/tpcds.csv'):
	lines = open(file_path, 'r').readlines()
	training_queries = []
	training_cards = []

	test_queries = []
	test_cards = []
	random.seed(42)

	single_table_query_in_train_prob = 0.0

	template2choice = {}
	q_count = 0
	single_table_q_count = 0
	max_num_preds = 0
	all_tables = []
	all_joins = []

	for line in lines:
		table2predicates = {} # (table_id, list_of_predicates)
		table2qreps = {}
		parts = line.strip().split('#')
		tables = parts[0].split(',')
		joins = parts[1].split(',')
		predicates = parts[2].split(',')
		card = int(parts[3])

		# ### test single tables
		# if len(tables) > 1:
		# 	continue
		# ########

		for join in joins:
			if join not in all_joins:
				all_joins.append(join)

		is_valid = True
		for t in tables:
			if t not in table2id:
				is_valid = False
				break
			table_id = table2id[t]
			table2predicates[table_id] = []
			table2qreps[table_id] = []
			if t not in all_tables:
				all_tables.append(t)

		if len(predicates) >= 3:
			# having predicate
			num_p = int(len(predicates)/3)

			if num_p > max_num_preds:
				max_num_preds = num_p

			for p_id in range(num_p):
				col = predicates[p_id * 3]
				op = predicates[p_id * 3 + 1]
				val = int(float(predicates[p_id * 3 + 2]))

				min_val = col2minmax[col][0]
				max_val = col2minmax[col][1]

				if val < min_val or val > max_val:
					is_valid = False
					break

				t = col.split('.')[0]
				table_id = table2id[t]

				col_id = col2id[col]
				global_col_id = global_col2id[col]
				if op not in ['=', 'IN', 'LIKE', '!=']:
					if op == '<':
						op = '<='
						nol_val = normalize_val(val,  min_val, max_val)
						table2predicates[table_id].append([col_id, [[0., nol_val]]])

					elif op == '>':
						op = '>='
						nol_val = normalize_val(val + 1, min_val, max_val)
						table2predicates[table_id].append([col_id, [[nol_val, 1.]]])

					elif op == '<=':
						nol_val = normalize_val(val + 1, min_val, max_val)
						table2predicates[table_id].append([col_id, [[0., nol_val]]])

					elif op == '>=':
						nol_val = normalize_val(val, min_val, max_val)
						table2predicates[table_id].append([col_id, [[nol_val, 1.]]])
				    
					table2qreps[table_id].append([col_id, op, [nol_val]])

				else: # op='='
					nol_val1 = normalize_val(val, min_val, max_val)
					nol_val2 = normalize_val(val + 1, min_val, max_val)

					table2predicates[table_id].append([col_id, [[nol_val1, nol_val2]]])

					table2qreps[table_id].append([col_id, '=', [nol_val1]])

		if is_valid:
			query_info = (table2predicates, table2qreps, sorted(tables), joins)

			if len(tables) == 5:
				training_queries.append(query_info)
				training_cards.append(card)
				test_queries.append(query_info)
				test_cards.append(card)
				q_count += 1

			else:
				if tuple(sorted(tables)) not in template2choice:
					# ran_number = random.uniform(0, 1)
					# if ran_number < 0.2:
					template2choice[tuple(sorted(tables))] = True
					# else:
					# 	template2choice[tuple(sorted(tables))] = False

				if template2choice[tuple(sorted(tables))]:
					training_queries.append(query_info)
					training_cards.append(card)
					# test_queries.append(table2predicates)
					# test_cards.append(card)
					q_count += 1
				# else:
					test_queries.append(query_info)
					test_cards.append(card)

			if q_count > num_q + test_size:
				break
			
	print('number of training qs')
	print(len(training_queries))
	template2queries = {}
	template2cards = {}

	for query_info, card in zip(training_queries, training_cards):
		table_list = query_info[-2]
		table_list = tuple(table_list)

		if table_list not in template2queries:
			template2queries[table_list] = [query_info]
			template2cards[table_list] = [card]
		else:
			template2queries[table_list].append(query_info)
			template2cards[table_list].append(card)

	test_template2queries = {}
	test_template2cards = {}
	
	print(len(test_queries))

	for query_info, card in zip(test_queries, test_cards):
		table_list = query_info[-2]
		table_list = tuple(table_list)

		if table_list not in test_template2queries:
			test_template2queries[table_list] = [query_info]
			test_template2cards[table_list] = [card]
		else:
			test_template2queries[table_list].append(query_info)
			test_template2cards[table_list].append(card)

	### shuffle training sets
	for table_list in template2queries:
		zipped = list(zip(template2queries[table_list], template2cards[table_list]))
		random.shuffle(zipped)

		new_qs, new_cards = zip(*zipped)	

		template2queries[table_list] = list(new_qs)
		template2cards[table_list] = list(new_cards)

	return template2queries, template2cards, test_template2queries, test_template2cards, max_num_preds, all_tables, all_joins





