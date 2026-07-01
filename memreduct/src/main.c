// Mem Reduct 鈥?ViewStage Memory Cleaner Subprocess (GUI stripped)

#include "routine.h"
#include "main.h"
#include "rapp.h"
#include "resource.h"

static INT g_exit_code = 0;

static INT _app_memoryclean(ULONG mask)
{
	ULONG privs[] = {
		SE_PROF_SINGLE_PROCESS_PRIVILEGE,
		SE_INCREASE_QUOTA_PRIVILEGE,
	};
	MEMORY_COMBINE_INFORMATION_EX combine_info_ex = {0};
	SYSTEM_FILECACHE_INFORMATION sfci = {0};
	SYSTEM_MEMORY_LIST_COMMAND command;
	NTSTATUS status;

	// Try enabling privileges first. If they're present (elevated admin or
	// SYSTEM), we can proceed directly. If not, need to elevate.
	// SYSTEM token has is_elevated=FALSE but contains these privileges,
	// so this check handles SYSTEM context correctly.
	if (!NT_SUCCESS(_r_sys_setprocessprivilege(NtCurrentProcess(), privs, RTL_NUMBER_OF(privs), TRUE)))
	{
		if (_r_app_runasadmin())
			return 0;

		return 1;
	}

	if (!mask)
		mask = _r_config_getulong(L"ReductMask2", REDUCT_MASK_DEFAULT);

	// Working set (vista+)
	if ((mask & REDUCT_WORKING_SET) == REDUCT_WORKING_SET)
	{
		command = MemoryEmptyWorkingSets;

		status = NtSetSystemInformation(SystemMemoryListInformation, &command, sizeof(SYSTEM_MEMORY_LIST_COMMAND));

		if (!NT_SUCCESS(status))
			_r_log(LOG_LEVEL_ERROR, NULL, L"NtSetSystemInformation", status, L"MemoryEmptyWorkingSets");
	}

	// System file cache
	if ((mask & REDUCT_SYSTEM_FILE_CACHE) == REDUCT_SYSTEM_FILE_CACHE)
	{
		sfci.MinimumWorkingSet = MAXSIZE_T;
		sfci.MaximumWorkingSet = MAXSIZE_T;

		status = NtSetSystemInformation(SystemFileCacheInformationEx, &sfci, sizeof(SYSTEM_FILECACHE_INFORMATION));

		if (!NT_SUCCESS(status))
			_r_log(LOG_LEVEL_ERROR, NULL, L"NtSetSystemInformation", status, L"SystemFileCacheInformation");
	}

	// Modified page list (vista+)
	if ((mask & REDUCT_MODIFIED_LIST) == REDUCT_MODIFIED_LIST)
	{
		command = MemoryFlushModifiedList;

		status = NtSetSystemInformation(SystemMemoryListInformation, &command, sizeof(SYSTEM_MEMORY_LIST_COMMAND));

		if (!NT_SUCCESS(status))
			_r_log(LOG_LEVEL_ERROR, NULL, L"NtSetSystemInformation", status, L"MemoryFlushModifiedList");
	}

	// Standby list (vista+)
	if ((mask & REDUCT_STANDBY_LIST) == REDUCT_STANDBY_LIST)
	{
		command = MemoryPurgeStandbyList;

		status = NtSetSystemInformation(SystemMemoryListInformation, &command, sizeof(SYSTEM_MEMORY_LIST_COMMAND));

		if (!NT_SUCCESS(status))
			_r_log(LOG_LEVEL_ERROR, NULL, L"NtSetSystemInformation", status, L"MemoryPurgeStandbyList");
	}

	// Standby priority-0 list (vista+)
	if ((mask & REDUCT_STANDBY_PRIORITY0_LIST) == REDUCT_STANDBY_PRIORITY0_LIST)
	{
		command = MemoryPurgeLowPriorityStandbyList;

		status = NtSetSystemInformation(SystemMemoryListInformation, &command, sizeof(SYSTEM_MEMORY_LIST_COMMAND));

		if (!NT_SUCCESS(status))
			_r_log(LOG_LEVEL_ERROR, NULL, L"NtSetSystemInformation", status, L"MemoryPurgeLowPriorityStandbyList");
	}

	// Flush registry cache (win8.1+)
	if (_r_sys_isosversiongreaterorequal(WINDOWS_8_1))
	{
		if ((mask & REDUCT_REGISTRY_CACHE) == REDUCT_REGISTRY_CACHE)
		{
			status = NtSetSystemInformation(SystemRegistryReconciliationInformation, NULL, 0);

			if (!NT_SUCCESS(status))
				_r_log(LOG_LEVEL_ERROR, NULL, L"NtSetSystemInformation", status, L"SystemRegistryReconciliationInformation");
		}
	}

	// Combine memory lists (win10+)
	if (_r_sys_isosversiongreaterorequal(WINDOWS_10))
	{
		if ((mask & REDUCT_COMBINE_MEMORY_LISTS) == REDUCT_COMBINE_MEMORY_LISTS)
		{
			status = NtSetSystemInformation(SystemCombinePhysicalMemoryInformation, &combine_info_ex, sizeof(MEMORY_COMBINE_INFORMATION_EX));

			if (!NT_SUCCESS(status))
				_r_log(LOG_LEVEL_ERROR, NULL, L"NtSetSystemInformation", status, L"SystemCombinePhysicalMemoryInformation");
		}
	}

	_r_config_setlong64(L"StatisticLastReduct", _r_unixtime_now());

	return 0;
}

static INT _app_skipuac_setup(void)
{
	// Enable skip UAC: creates a scheduled task so _r_app_runasadmin()
	// can elevate without UAC prompt
	if (!_r_sys_iselevated())
	{
		if (_r_app_runasadmin())
			return 0;

		return 1;
	}

	_r_skipuac_enable(NULL, TRUE);

	return 0;
}

static INT _app_uninstall(void)
{
	if (!_r_sys_iselevated())
	{
		if (_r_app_runasadmin())
			return 0;

		return 1;
	}

	_r_skipuac_enable(NULL, FALSE);

	return 0;
}

static BOOLEAN NTAPI _app_parseargs(R_CMDLINE_INFO_CLASS type)
{
	PCWSTR cmdline = _r_sys_getcommandline();

	switch (type)
	{
		case CmdlineClean:
		{
			PR_STRING clean_args;

			_r_sys_getopt(cmdline, L"clean", &clean_args);

			if (!clean_args)
				break;

			ULONG mask = 0;

			if (_r_str_isequal2(&clean_args->sr, L"full", TRUE))
				mask = REDUCT_MASK_ALL;

			_r_obj_dereference(clean_args);

			if (!mask)
				mask = REDUCT_MASK_DEFAULT;

			g_exit_code = _app_memoryclean(mask);

			return TRUE;
		}

	case CmdlineHelp:
		{
			g_exit_code = 0;

			OutputDebugStringW(L"Mem Reduct - ViewStage Memory Cleaner Subprocess\n");
			OutputDebugStringW(L"\n");
			OutputDebugStringW(L"Usage:\n");
			OutputDebugStringW(L"  memreduct-viewstage.exe --clean[:full]\n");
			OutputDebugStringW(L"  memreduct-viewstage.exe --skipuac\n");
			OutputDebugStringW(L"  memreduct-viewstage.exe --check-skipuac\n");
			OutputDebugStringW(L"  memreduct-viewstage.exe --uninstall\n");
			OutputDebugStringW(L"  memreduct-viewstage.exe --help\n");
			OutputDebugStringW(L"\n");
			OutputDebugStringW(L"Commands:\n");
			OutputDebugStringW(L"  --clean          Clean memory (default regions)\n");
			OutputDebugStringW(L"  --clean:full     Clean all memory regions\n");
			OutputDebugStringW(L"  --skipuac        Enable skip-UAC (no UAC prompt on next clean)\n");
			OutputDebugStringW(L"  --check-skipuac  Check if skip-UAC task is installed\n");
			OutputDebugStringW(L"  --uninstall      Remove skip-UAC and legacy scheduled tasks\n");

			return TRUE;
		}
	}

	// Check for --skipuac
	PR_STRING skip_args = NULL;

	_r_sys_getopt(cmdline, L"skipuac", &skip_args);

	if (skip_args)
	{
		_r_obj_dereference(skip_args);

		g_exit_code = _app_skipuac_setup();

		return TRUE;
	}

	// Check for --uninstall
	PR_STRING uninstall_args = NULL;

	_r_sys_getopt(cmdline, L"uninstall", &uninstall_args);

	if (uninstall_args)
	{
		_r_obj_dereference(uninstall_args);

		g_exit_code = _app_uninstall();

		return TRUE;
	}

	return FALSE;
}

INT APIENTRY wWinMain(
	_In_ HINSTANCE hinst,
	_In_opt_ HINSTANCE prev_hinst,
	_In_ LPWSTR cmdline,
	_In_ INT show_cmd
)
{
	PCWSTR cmdline_full = _r_sys_getcommandline();

	// Handle custom args that _r_app_initialize doesn't recognize
	if (_r_sys_getopt(cmdline_full, L"skipuac", NULL))
	{
		if (!_r_sys_iselevated())
		{
			// Re-launch as admin via ShellExecuteExW(runas) WITHOUT
			// SEE_MASK_FLAG_NO_UI so user always sees the UAC prompt
			SHELLEXECUTEINFOW shex = {0};

			shex.cbSize = sizeof(shex);
			shex.fMask = SEE_MASK_UNICODE | SEE_MASK_NOZONECHECKS;
			shex.lpVerb = L"runas";
			shex.lpFile = _r_sys_getimagepath();
			shex.lpParameters = L"--skipuac";
		shex.nShow = SW_HIDE;

		if (ShellExecuteExW(&shex))
			return 0;

		return 1;
	}

	CoInitializeEx(NULL, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);

	_r_skipuac_enable(NULL, TRUE);

		CoUninitialize();

		return 0;
	}

	if (_r_sys_getopt(cmdline_full, L"clean", NULL))
	{
		if (!_r_sys_iselevated())
		{
			CoInitializeEx(NULL, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);

			BOOLEAN ok = _r_skipuac_run();

			CoUninitialize();

			return ok ? 0 : 1;
		}

		// Elevated — parse mask and run clean directly
		PR_STRING clean_args = NULL;

		_r_sys_getopt(cmdline_full, L"clean", &clean_args);

		ULONG mask = 0;

		if (clean_args)
		{
			if (_r_str_isequal2(&clean_args->sr, L"full", TRUE))
				mask = REDUCT_MASK_ALL;
			else if (clean_args->sr.length > 0)
				mask = wcstoul(clean_args->buffer, NULL, 10);

			_r_obj_dereference(clean_args);
		}

		if (!mask)
			mask = REDUCT_MASK_DEFAULT;

		_r_app_getprofiledirectory();

		return _app_memoryclean(mask);
	}

	if (_r_sys_getopt(cmdline_full, L"check-skipuac", NULL))
	{
		CoInitializeEx(NULL, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);

		INT ret = _r_skipuac_isenabled() ? 0 : 1;

		CoUninitialize();

		return ret;
	}

	if (_r_sys_getopt(cmdline_full, L"uninstall", NULL))
	{
		if (!_r_sys_iselevated())
		{
			SHELLEXECUTEINFOW shex = {0};

			shex.cbSize = sizeof(shex);
			shex.fMask = SEE_MASK_UNICODE | SEE_MASK_NOZONECHECKS;
			shex.lpVerb = L"runas";
			shex.lpFile = _r_sys_getimagepath();
			shex.lpParameters = L"--uninstall";
			shex.nShow = SW_HIDE;

			if (ShellExecuteExW(&shex))
				return 0;

			return 1;
		}

		CoInitializeEx(NULL, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);

		_r_skipuac_enable(NULL, FALSE);

		CoUninitialize();

		return 0;
	}

	if (!_r_app_initialize(&_app_parseargs))
		return g_exit_code;

	return 1;
}


